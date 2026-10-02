import test from 'node:test';
import assert from 'node:assert/strict';
import { CargoGame, CARGO_SHAPE_GROUPS } from '../src/projects/cargoEngine.js';
import { TournamentGame, cutsOffCorner } from '../src/projects/engine.js';
import { orientSeal, unorientSeal } from '../src/projects/sealOrientation.js';
import { MatchRuntime } from '../src/projects/matchRuntime.js';
import { ALL_PROJECTS } from '../src/projects/catalog.js';

const family = shape => CARGO_SHAPE_GROUPS.findIndex(group => group.includes(shape));
const sealProject = ALL_PROJECTS.find(p => p.sealEveryMoves);
const cargoProject = ALL_PROJECTS.find(p => p.cargoTransport);
const canonical = game => game.sealedCells.map(i => unorientSeal(i, 4, game.sealOrientation)).sort((a,b) => a-b);

test('cargo shares the unchanged family stream but uses reproducible per-side variants', () => {
  let differences = 0, matches = 0;
  for (let seed = 0; seed < 100; seed += 1) {
    const yellow = new CargoGame(cargoProject, { seed, side:'yellow' });
    const white = new CargoGame(cargoProject, { seed, side:'white' });
    const legacy = new CargoGame(cargoProject, { seed });
    const repeat = new CargoGame(cargoProject, { seed, side:'yellow' });
    for (let n = 0; n < 30; n += 1) {
      const a = yellow.nextCargo().shape, b = white.nextCargo().shape;
      assert.equal(a, repeat.nextCargo().shape);
      assert.equal(family(a), family(b));
      assert.equal(family(a), family(legacy.nextCargo().shape));
      assert.equal(yellow.shapeState, white.shapeState);
      if (CARGO_SHAPE_GROUPS[family(a)].length > 1) a === b ? matches++ : differences++;
      yellow.spawnNumber(); // Faster numeric play must not change future families.
    }
  }
  assert.ok(differences > 0 && matches > 0, 'independent, not forced-opposite variants');
});

test('all eight seal symmetries are reversible and preserve corner legality', () => {
  for (let orientation = 0; orientation < 8; orientation += 1) {
    for (let index = 0; index < 16; index += 1) assert.equal(unorientSeal(orientSeal(index,4,orientation),4,orientation), index);
    for (let a=0;a<16;a++) for(let b=a+1;b<16;b++) for(let c=b+1;c<16;c++) {
      const cells=[a,b,c];
      assert.equal(cutsOffCorner(cells,4,4), cutsOffCorner(cells.map(i=>orientSeal(i,4,orientation)),4,4));
    }
  }
});

test('seal patterns remain shared across every round, without repeats or forbidden corners', () => {
  let differences=0;
  for(let seed=0;seed<100;seed++) {
    const yellow=new TournamentGame(sealProject,{seed,side:'yellow'});
    const white=new TournamentGame(sealProject,{seed,side:'white'});
    const legacy=new TournamentGame(sealProject,{seed});
    for(let round=0;round<30;round++) {
      assert.deepEqual(canonical(yellow), canonical(white));
      assert.deepEqual(canonical(yellow), legacy.sealedCells);
      assert.equal(yellow.sealState,white.sealState);
      if(JSON.stringify(yellow.sealedCells)!==JSON.stringify(white.sealedCells)) differences++;
      for(const game of [yellow,white,legacy]) {
        assert.equal(game.sealedCells.length,3);
        assert.equal(cutsOffCorner(game.sealedCells,4,4),false);
        const previous=game.sealedCells.slice();
        game.rotateSeals();
        assert.ok(game.sealedCells.every(i=>!previous.includes(i)));
      }
    }
  }
  assert.ok(differences>0);
});

test('new variant state survives checkpoints; old checkpoints keep legacy behavior', () => {
  for(const project of [cargoProject,sealProject]) {
    const bootstrap={instance_id:'variants',project_ref:project.id,rules_version:project.adapterRulesVersion,seed:'resume-variants',side:'yellow',sequence:0};
    const runtime=new MatchRuntime(bootstrap,{now:()=>0});
    for(let n=0;n<7;n++) project.cargoTransport ? runtime.game.nextCargo() : runtime.game.rotateSeals();
    const checkpoint=runtime.checkpoint();
    const resumed=new MatchRuntime({...bootstrap,checkpoint},{now:()=>0});
    for(let n=0;n<10;n++) {
      if(project.cargoTransport) assert.equal(runtime.game.nextCargo().shape,resumed.game.nextCargo().shape);
      else assert.deepEqual(runtime.game.rotateSeals(),resumed.game.rotateSeals());
    }
    const legacy=new MatchRuntime({...bootstrap,side:'solo'},{now:()=>0});
    const old=legacy.checkpoint();delete old.state.cargoVariantState;delete old.state.sealOrientation;
    const oldResumed=new MatchRuntime({...bootstrap,checkpoint:old},{now:()=>0});
    for(let n=0;n<10;n++) {
      if(project.cargoTransport) assert.equal(legacy.game.nextCargo().shape,oldResumed.game.nextCargo().shape);
      else assert.deepEqual(legacy.game.rotateSeals(),oldResumed.game.rotateSeals());
    }
  }
});
