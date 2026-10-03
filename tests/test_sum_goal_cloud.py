"""Cloud goal identity, masking and native reader regressions."""
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from Config import pattern_catalog
from engine_core.GoalSpec import GoalSpec
from engine_core.BookReader import BookReader, BookReaderDispatcher
from engine_core.BookReaderEX import BookReaderEX
from engine_core.BookReaderBC import BookReaderBC
from engine_core.VBoardMover import decode_board, encode_board
from backend.analysis import normalize_target_value
from backend.lookup_mask import lookup_key
from backend import tablebase_catalog


def variant_board(pattern, tiles):
    board = decode_board(np.uint64(pattern_catalog[pattern]['seed_boards'][0])).copy()
    positions = np.flatnonzero(board.ravel() != 32768)
    board.ravel()[positions] = 0
    board.ravel()[positions[:len(tiles)]] = tiles
    return board


def test_goal_identity_and_mask():
    assert normalize_target_value('sum-1800') == ('sum-1800', 'sum-1800', 'sum-1800')
    for token in ['sum-3', 'sum-901', 'sum-16384']:
        with pytest.raises(ValueError): GoalSpec.parse(token)
    board = variant_board('3x3', [1024, 512, 256, 4, 2])
    encoded = int(encode_board(board))
    assert lookup_key(encoded, '3x3_sum-1800') == f'{encoded:016x}'
    assert lookup_key(encoded, '3x3_1024') != f'{encoded:016x}'
    assert GoalSpec.parse('sum-1800').reached(encoded)
    assert not GoalSpec.parse('sum-1800').reached(encode_board(variant_board('3x3',[16384,2])))


@pytest.mark.parametrize('pattern,target,tiles', [('3x3',1800,[1024,512,256,4,2]),('2x4',900,[512,256,128,2])])
@pytest.mark.parametrize('algorithm', ['classic','ex','bc'])
def test_native_terminal_without_layer_files(tmp_path,pattern,target,tiles,algorithm):
    token=f'sum-{target}'; full=f'{pattern}_{token}'; board=variant_board(pattern,tiles)
    if algorithm=='classic':
        results,dtype=BookReader.move_on_dic(board,pattern,token,full,[(str(tmp_path),'float64')])
    else:
        reader=(BookReaderEX if algorithm=='ex' else BookReaderBC)(pattern,token)
        results,dtype=reader.move_on_dic(board,full,[(str(tmp_path),'float64')])
    assert any(value == 1 for value in results.values()), results
    assert all(value in (None,1) for value in results.values()), results
    if algorithm!='bc':
        if algorithm=='classic': results,dtype=BookReader.move_on_dic(board,pattern,token,full,[(str(tmp_path),'1-float64')])
        else: results,dtype=reader.move_on_dic(board,full,[(str(tmp_path),'1-float64')])
        assert any(value == 0 for value in results.values()), results


def test_catalog_goal_metadata_and_ai_exclusion():
    entry=dict(pattern='3x3',target='sum-1800',dtype='float64',spawn_rate=.1,_full_pattern='3x3_sum-1800',_provider='local')
    with patch.object(tablebase_catalog,'_iter_available_entries',return_value=[entry]):
        result=tablebase_catalog.get_available_tablebases()[0]
        assert result['goal']==dict(kind='sum',value=1800)
        assert not result['ai']['compatible']
        assert tablebase_catalog.get_catalog_target_tiles()==['sum-1800']


@pytest.mark.parametrize('pattern,target,tiles', [('3x3',1800,[1024,512,128,64,32,16,8,4,2]),('2x4',900,[512,256,64,32,16,8,4,2])])
def test_real_ex_tables(pattern,target,tiles):
    root=os.environ.get('SUM_GOAL_TABLE_ROOT')
    if not root: pytest.skip('Set SUM_GOAL_TABLE_ROOT to test the computed EX tables')
    full=f'{pattern}_sum-{target}'; path=Path(root)/f'{full}_ex'
    assert path.is_dir()
    reader=BookReaderDispatcher(); paths=[(str(path),'float64')]
    reader.dispatch(paths,pattern,f'sum-{target}')
    assert reader.use_ex
    # Below the boundary: a full, unmergeable board must not become an automatic win.
    board=variant_board(pattern,tiles)
    results,dtype=reader.move_on_dic(board,pattern,f'sum-{target}',full)
    assert not any(value==1 for value in results.values()),results
    initial=variant_board(pattern,[2,2])
    results,dtype=reader.move_on_dic(initial,pattern,f'sum-{target}',full)
    assert dtype=='float64'
    assert any(isinstance(value,float) and 0 < value <= 1 for value in results.values()),results
    print(full, results)

def test_sum_analysis_records_first_and_last_step(tmp_path):
    from backend.analysis_core import Analyzer
    from types import SimpleNamespace
    from engine_core.VBoardMover import s_move_board
    from engine_core.replay_utils import replay_sentinel
    board=variant_board('3x3',[1024,512,256,4,2])
    encoded=encode_board(board)
    moved,_=s_move_board(encoded,4)
    assert int(moved)!=int(encoded)
    after=decode_board(moved).copy()
    spawn=int(np.flatnonzero(after.ravel()==0)[0]); after.ravel()[spawn]=2
    terminal=encode_board(after)
    records=np.zeros(2,dtype='uint64,uint8,uint32,uint32,uint32,uint32')
    records[0]=(encoded,(3<<5)|(spawn<<1),4_000_000_000,4_000_000_000,4_000_000_000,4_000_000_000)
    records[1]=replay_sentinel(terminal)
    source=tmp_path/'3x3_sum-1800_input.rpl'; records.tofile(source)
    decoded=SimpleNamespace(decode=lambda:None, record_list=[(encoded,0,4,1,spawn),(terminal,0,1,1,0)],variant='3x3',final_score=0)
    with patch('backend.analysis_core.ReplayDecoder',return_value=decoded), patch('backend.analysis_core.resolve_configured_tablebase',return_value=None), patch('backend.analysis_core.build_filepath_map_entry',return_value=[(str(tmp_path),'float64')]):
        analyzer=Analyzer(str(source),'3x3','sum-1800','3x3_sum-1800',str(tmp_path))
    analyzer.generate_reports()
    assert analyzer.sum_goal_completed
    assert len(analyzer.segment_summaries)==1
    segment=analyzer.segment_summaries[0]
    assert segment['evaluated_moves']==1
    assert sum(segment['performance_counts'].values())==1
    output=np.fromfile(tmp_path/segment['replay_file'],dtype=records.dtype)
    assert len(output)==2
    assert int(output['f0'][1])==int(terminal)
    assert segment['end_index']==1  # Any subsequent archived moves are excluded.


def test_sum_analysis_keeps_early_certain_steps(tmp_path):
    from backend.analysis_core import Analyzer
    from types import SimpleNamespace
    analyzer=Analyzer.__new__(Analyzer)
    analyzer.goal=GoalSpec.parse('sum-1800'); analyzer.sum_goal_completed=False
    analyzer.variant='3x3'; analyzer.pattern='3x3'; analyzer.full_pattern='3x3_sum-1800'
    analyzer.bm=__import__('engine_core.BoardMover',fromlist=['decode_board'])
    analyzer.book_reader=SimpleNamespace(move_on_dic=lambda *args: ({'left':1.,'right':1.,'up':None,'down':1.},'float64'))
    analyzer.record=np.zeros(4000,dtype='uint64,uint8,uint32,uint32,uint32,uint32')
    analyzer.clear_analysis()
    board=variant_board('3x3',[2,2])
    assert analyzer._analyze_one_step(board,board,'Left',1,1)
    assert analyzer.step_count==1 and analyzer.rec_step_count==1
    assert not analyzer.sum_goal_completed

@pytest.mark.parametrize('algorithm', ['classic','ex','bc'])
@pytest.mark.parametrize('pattern,target,tiles', [('3x3',1800,[8,1024,512,128,8,64,32,16,8]),('2x4',900,[8,512,256,64,32,16,4,8])])
def test_terminal_sum_still_requires_a_valid_move(tmp_path,algorithm,pattern,target,tiles):
    token=f'sum-{target}'; full=f'{pattern}_{token}'; board=variant_board(pattern,tiles)
    assert GoalSpec.parse(token).reached(encode_board(board))
    if algorithm=='classic':
        results,_=BookReader.move_on_dic(board,pattern,token,full,[(str(tmp_path),'float64')])
    else:
        reader=(BookReaderEX if algorithm=='ex' else BookReaderBC)(pattern,token)
        results,_=reader.move_on_dic(board,full,[(str(tmp_path),'float64')])
    assert all(value in (None, "") for value in results.values()), results


@pytest.mark.parametrize('mode_name,class_name',[('goodness','GoodnessBattleMode'),('free_goodness','FreeGoodnessBattleMode')])
def test_battle_rejects_sum_catalog_entry(mode_name,class_name):
    import importlib
    module=importlib.import_module(f'backend.battle.modes.{mode_name}.mode')
    with patch.object(module,'resolve_tablebase',return_value={'pattern':'3x3','target':'sum-1800'}):
        with pytest.raises(ValueError,match='table_unavailable'):
            getattr(module,class_name)().validate_settings({'full_pattern':'3x3_sum-1800'})


def test_actual_manifest_discovers_both_local_directories():
    root=os.environ.get('SUM_GOAL_TABLE_ROOT')
    if not root: pytest.skip('Set SUM_GOAL_TABLE_ROOT to test local catalog discovery')
    with patch.dict(os.environ,{'CLOUD_TABLEBASE_ROOT':root}):
        entries=tablebase_catalog._iter_local_entries()
    found={entry['_full_pattern']:entry for entry in entries}
    for full in ('3x3_sum-1800','2x4_sum-900'):
        assert found[full]['dtype']=='float64'
        assert found[full]['_provider']=='local'
