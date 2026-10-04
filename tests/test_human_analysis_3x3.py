import copy
import json
import math

import pytest

from backend.human_play.analysis_grade import (
    ACCURACY_3X3, COMBO_3X3, PERFECT_3X3, GRADE_VERSION,
    THREE_BY_THREE_PROFILE, grade_result, points_3x3, time_bonus_3x3,
)
from backend.human_play.analysis_summary import (
    build_summary, prepare_poster_summary, get_summary, list_summaries, save_summary,
)


@pytest.mark.parametrize('thresholds', [COMBO_3X3, PERFECT_3X3, ACCURACY_3X3])
def test_each_anchor_and_midpoint_interpolates(thresholds):
    assert list(thresholds) == sorted(set(thresholds))
    assert points_3x3(thresholds[0] / 2, thresholds) == 0
    assert points_3x3(thresholds[-1] + 1, thresholds) == 9
    for index, threshold in enumerate(thresholds, 1):
        assert points_3x3(threshold, thresholds) == index
    for index, (left, right) in enumerate(zip(thresholds, thresholds[1:]), 1):
        assert points_3x3((left + right) / 2, thresholds) == pytest.approx(index + .5)


@pytest.mark.parametrize('total,seconds,points', [
    (1023, 271, 0), (1023, 270, 1), (1023, 232.5, 1.5),
    (1023, 195, 2), (1023, 157.5, 2.5), (1023, 120, 3), (1023, 1, 3),
    (1024, 661, 0), (1024, 660, 1), (1024, 570, 1.5),
    (1535, 480, 2), (1535, 390, 2.5), (1535, 300, 3),
    (1536, 99999, 3), (1790, None, 3), (1000, None, 0),
    (1000, 0, 0), (1000, float('nan'), 0),
])
def test_time_bonus_uses_whole_game_time_and_inclusive_progress_boundaries(total, seconds, points):
    assert time_bonus_3x3(total, None if seconds is None else seconds * 1000) == points


def segment(start, end, fit, perfect, other, combo):
    return dict(start_index=start, end_index=end, evaluated_moves=end-start,
                goodness_of_fit=fit, max_combo=combo,
                performance_counts={'Perfect!': perfect, 'Mistake!': other})


def prepare(summary, target='sum-1790', board_sum=1200):
    return prepare_poster_summary(summary, pattern='3x3', target=target, variant='3x3',
                                  run_state={'board': [board_sum]+[0]*8, 'elapsed': 600000})


@pytest.mark.parametrize('target', ['1024', 'sum-1790'])
def test_geometric_mean_counts_only_categorized_moves_not_skipped_warmup(target):
    summary = build_summary([
        segment(4, 28, .999**20, 16, 4, 13),
        segment(32, 46, .99**10, 7, 3, 7),
    ], [10000]*50)
    original = copy.deepcopy(summary)
    result = prepare(summary, target)['aggregate']
    assert result['evaluated_moves'] == 30
    assert result['mean_single_step_accuracy'] == pytest.approx(math.exp((20*math.log(.999)+10*math.log(.99))/30))
    assert result['perfect_rate'] == pytest.approx(23/30)
    assert result['stage_count'] == 2  # Short valid 3x3 endgames are not discarded.
    assert result['run_elapsed_ms'] == 500000  # Includes skipped / unanalysed moves.
    assert result['poster_eligible']
    assert summary == original


def test_zero_fit_failure_is_eligible_and_missing_timing_is_not_zero_seconds():
    summary = build_summary([segment(0, 24, 0, 1, 19, 1)], [None]*24)
    aggregate = prepare(summary, board_sum=1000)['aggregate']
    assert aggregate['poster_eligible']
    assert aggregate['mean_single_step_accuracy'] == 0
    assert aggregate['run_elapsed_ms'] is None
    assert grade_result(variant='3x3', goal_tile=None, score=1000, aggregate=aggregate) == (0, 'F')


def test_partial_or_long_timing_cannot_create_fast_bonus():
    partial = prepare(build_summary([segment(0, 20, .99, 18, 2, 17)], [1000]*19+[None]))
    assert partial['aggregate']['run_elapsed_ms'] is None
    long = prepare(build_summary([segment(0, 20, .99, 18, 2, 17)], [1000]*19+[1500000]))
    assert long['aggregate']['run_elapsed_ms'] == 1519000


def test_fractional_points_are_added_without_rounding():
    aggregate = dict(grading_profile=THREE_BY_THREE_PROFILE, mean_single_step_accuracy=.99595,
                     perfect_rate=.65, max_combo=15, evaluated_moves=100,
                     run_board_sum=900, run_elapsed_ms=232500)
    points, grade = grade_result(variant='3x3', goal_tile=None, score=1000, aggregate=aggregate)
    assert points == pytest.approx(6)
    assert grade == 'C'
    aggregate['max_combo'] = 290
    aggregate['perfect_rate'] = 1
    aggregate['mean_single_step_accuracy'] = 1
    aggregate['evaluated_moves'] = 500
    aggregate['run_board_sum'] = 1536
    assert grade_result(variant='3x3', goal_tile=None, score=1, aggregate=aggregate) == (30, 'X')


def test_unsupported_targets_and_variants_are_not_enabled():
    summary = build_summary([segment(0, 20, 1, 20, 0, 20)], [1000]*20)
    for pattern, target, variant in [('3x3', '512', '3x3'), ('2x4', 'sum-894', '2x4')]:
        result = prepare_poster_summary(summary, pattern=pattern, target=target, variant=variant)
        assert 'grading_profile' not in result['aggregate']


@pytest.mark.parametrize('combo,perfect,accuracy,expected', [
    (26, 1, 0, 'A'), (27, 1, 0, 'S'),
    (185, 1, 0, 'S'), (186, 1, 0, 'SS'),
    (289, 1, 0, 'SS'), (290, 1, 0, 'SSS'),
    (290, 1, ACCURACY_3X3[1] - 1e-10, 'SSS'),
    (290, 1, ACCURACY_3X3[1], 'SSS'),
    (290, 1, ACCURACY_3X3[2] - 1e-10, 'SSS'),
    (290, 1, ACCURACY_3X3[2], 'X'),
])
def test_new_top_grade_boundaries_without_timer(combo, perfect, accuracy, expected):
    aggregate = dict(grading_profile=THREE_BY_THREE_PROFILE,
                     mean_single_step_accuracy=accuracy, perfect_rate=perfect,
                     max_combo=combo, evaluated_moves=500,
                     run_board_sum=1016, run_elapsed_ms=None)
    assert grade_result(variant='3x3', goal_tile=None, score=6948,
                        aggregate=aggregate)[1] == expected


def test_reference_no_timer_game_is_now_ss():
    aggregate = dict(grading_profile=THREE_BY_THREE_PROFILE,
                     mean_single_step_accuracy=.9999864, perfect_rate=406/454,
                     max_combo=194, evaluated_moves=454,
                     run_board_sum=1016, run_elapsed_ms=None)
    points, grade = grade_result(variant='3x3', goal_tile=None, score=6948,
                                aggregate=aggregate)
    assert points == pytest.approx(17.7044830267)
    assert grade == 'SS'


@pytest.mark.parametrize('combo,expected_grade', [(26, 'E'), (27, 'D'), (28, 'D')])
def test_grade_boundaries_use_unrounded_total(combo, expected_grade):
    aggregate = dict(grading_profile=THREE_BY_THREE_PROFILE, mean_single_step_accuracy=0,
                     perfect_rate=0, max_combo=combo, evaluated_moves=100,
                     run_board_sum=900, run_elapsed_ms=None)
    points, grade = grade_result(variant='3x3', goal_tile=None, score=1000, aggregate=aggregate)
    assert grade == expected_grade
    if combo == 26:
        assert points == pytest.approx(2.9)


@pytest.mark.parametrize('target', ['1024', 'sum-1790'])
@pytest.mark.parametrize('old_version', [2, 3])
def test_historical_summary_gets_button_and_grade_without_reanalysis(tmp_path, monkeypatch, target, old_version):
    from backend.auth.db import init_auth_db, auth_db
    from backend.human_play import leaderboards
    from backend.human_play.store import init_db, database
    monkeypatch.setenv('CLOUD_AUTH_DB', str(tmp_path/'auth.sqlite3'))
    monkeypatch.setenv('HUMAN_PLAY_DB', str(tmp_path/'human.sqlite3'))
    init_auth_db(); init_db()
    with auth_db() as db:
        db.execute("""INSERT INTO users(id,email,password_hash,display_name,status,created_at,updated_at)
            VALUES(1,'poster@example.invalid','!','Player','active','2026-01-01','2026-01-01')""")
    state = dict(score=15000, board=[1024,512,128,64,32,16,8,4,2], seq=300, elapsed=600000)
    with database() as db:
        db.execute("""INSERT INTO human_runs(id,user_id,browser,variant,request_id,seed,threshold,status,
            created,ended,reason,writer,state,archive,visible,has_replay)
            VALUES('3x3-run',1,'browser','3x3','request','seed',0,'sealed',1,2,'game_over','writer',?,X'01',1,1)""",
            (json.dumps(state),))
    saved = build_summary([segment(0,300,.999999**300,300,0,300)], [2000]*300)
    sid = save_summary(run_id='3x3-run', user_id=1, pattern='3x3', target=target, job_id='job', summary=saved)
    # Simulate a paid summary from before 3x3 posters existed.
    with database() as db:
        old = json.loads(db.execute('SELECT summary_json FROM human_analysis_summaries WHERE id=?',(sid,)).fetchone()[0])
        old['aggregate'] = saved['aggregate']
        old['aggregate']['poster_eligible'] = False
        old.pop('run_timing')
        old['grade_version'] = old_version; old['grade'] = 'SSS' if old_version == 3 else None
        db.execute('UPDATE human_analysis_summaries SET summary_json=?,aggregate_json=? WHERE id=?',
                   (json.dumps(old),json.dumps(old['aggregate']),sid))
    assert list_summaries('3x3-run',1)[0]['aggregate']['poster_eligible']
    result = get_summary(sid,1)
    assert result['grade_version'] == GRADE_VERSION
    assert result['grade'] == 'X'
    assert result['aggregate']['run_elapsed_ms'] == 600000
    assert result['aggregate']['mean_single_step_accuracy'] == pytest.approx(.999999)
    with database() as db:
        assert db.execute('SELECT COUNT(*) FROM human_analysis_summaries').fetchone()[0] == 1
        indexed = db.execute('SELECT grade,grade_version FROM human_analysis_results WHERE summary_id=?', (sid,)).fetchone()
        assert indexed['grade'] == 'X'
        assert indexed['grade_version'] == GRADE_VERSION
        board = leaderboards.catalog(db)['strength']
        assert board == [{'variant':'3x3','pattern':'3x3','target':target,'results':1}]
