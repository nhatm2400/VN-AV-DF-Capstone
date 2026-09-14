"""Exercise the actual embedded player handlers with queued media events in Node."""
import shutil
import subprocess
import unittest
from pathlib import Path


class PlayerSyncTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node required for player event test')
    def test_two_way_seek_play_pause_rate_and_missing_roi(self):
        source = (Path(__file__).resolve().parents[2] / 'src/tools/review/clip_review.py').read_text('utf-8')
        script = source.split('<script>', 1)[1].split('</script>', 1)[0]
        # Parse the entire JS script, execute only the synchronization setup.
        setup = script.split('function fmt', 1)[0]
        harness = r'''
const assert=require('node:assert/strict');
const queue=[];
class Media {
  constructor(){this.handlers={};this.time=0;this.paused=true;this.readyState=1;this.duration=5;this.rate=1;this.seeking=false;}
  addEventListener(e,fn){(this.handlers[e]??=[]).push(fn);}
  emit(e){queue.push(()=>{for(const fn of this.handlers[e]||[])fn();});}
  get currentTime(){return this.time;}
  set currentTime(t){this.time=t;this.seeking=true;this.emit('seeking');queue.push(()=>{this.seeking=false;});this.emit('seeked');}
  get playbackRate(){return this.rate;}
  set playbackRate(r){this.rate=r;this.emit('ratechange');}
  play(){if(this.paused){this.paused=false;this.emit('play');}return Promise.resolve();}
  pause(){if(!this.paused){this.paused=true;this.emit('pause');}}
}
const media={vid:new Media(),roivid:new Media()};
const document={getElementById:id=>media[id]};
function flush(){let n=0;while(queue.length){assert.ok(n++<100,'Event feedback loop');queue.shift()();}}
'''
        checks = r'''
S.clips=[{has_roi:true}];
originalVideo.currentTime=2;flush();assert.equal(roiVideo.currentTime,2);
roiVideo.currentTime=1;flush();assert.equal(originalVideo.currentTime,1);
originalVideo.play();flush();assert.equal(roiVideo.paused,false);
roiVideo.pause();flush();assert.equal(originalVideo.paused,true);
roiVideo.playbackRate=0.5;flush();assert.equal(originalVideo.playbackRate,0.5);
originalVideo.play();flush();roiVideo.time=0;originalVideo.emit('timeupdate');flush();assert.equal(roiVideo.time,originalVideo.time);
roiVideo.readyState=0;originalVideo.currentTime=3;flush();roiVideo.readyState=1;roiVideo.emit('loadedmetadata');flush();assert.equal(roiVideo.time,3);
originalVideo.pause();flush();S.clips[0].has_roi=false;originalVideo.currentTime=4;originalVideo.play();flush();assert.equal(roiVideo.time,3);assert.equal(roiVideo.paused,true);
console.log('Player synchronization: PASS');
'''
        program = 'new Function(' + __import__('json').dumps(script) + ');\n' + harness + setup + checks
        result = subprocess.run(['node'], input=program, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
