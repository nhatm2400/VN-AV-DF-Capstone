"""Exercise the embedded JS: dialog, saves, navigation and error handling."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


class RejectDialogTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node required')
    def test_reason_popup_save_cancel_failure_and_keep(self):
        source = (Path(__file__).resolve().parents[2] / 'src/tools/review/clip_review.py').read_text('utf-8')
        script = source.split('<script>', 1)[1].split('</script>', 1)[0].rsplit('load();', 1)[0]
        harness = r'''
const assert=require('node:assert/strict');
const elements={};const handlers={};const requests=[];
function element(){return {style:{},dataset:{},handlers:{},open:false,checked:false,paused:true,
  addEventListener(k,fn){this.handlers[k]=fn;},pause(){this.paused=true;},
  play(){this.paused=false;return Promise.resolve();},load(){},removeAttribute(){},
  showModal(){this.open=true;},close(){this.open=false;},querySelectorAll(){return [];}};}
const document={getElementById(id){return elements[id]??=element();},
  addEventListener(k,fn){handlers[k]=fn;}};
let succeed=true;let release=null;let hold=false;
async function fetch(url,options){requests.push(JSON.parse(options.body));
  if(hold)await new Promise(resolve=>release=resolve);
  return {ok:succeed,json:async()=>({ok:succeed,reason:requests.at(-1).reason,bad_intervals:[]})};}
const alerts=[];function alert(message){alerts.push(message);}
'''
        checks = r'''
(async()=>{
S.clips=[{clip_id:'a',has_roi:true},{clip_id:'b',has_roi:true},{clip_id:'c',has_roi:false}];
S.reasons=[['mouth','Miệng'],['cut','Chuyển cảnh']];
const dialog=document.getElementById('rejectDialog');
openReject();assert.equal(dialog.open,true);closeReject();assert.equal(requests.length,0);
openReject();handlers.keydown({target:{tagName:'BODY'},key:'k'});assert.equal(requests.length,0);
hold=true;const saving=mark('reject','mouth');await Promise.resolve();
await mark('reject','cut');go(1);assert.equal(requests.length,1);assert.equal(S.i,0);
release();await saving;hold=false;
assert.equal(S.i,1);assert.equal(S.clips[0].reason,'mouth');assert.equal(dialog.open,false);
assert.deepEqual(requests[0],{i:0,decision:'reject',reason:'mouth',bad_intervals:[]});
await mark('keep');assert.equal(S.i,1);assert.equal(S.clips[1].dec,'keep');
document.getElementById('auto').checked=true;await mark('keep');assert.equal(S.i,2);
succeed=false;openReject();await mark('reject','cut');assert.equal(S.i,2);
assert.equal(dialog.open,true);assert.equal(S.clips[2].dec,undefined);assert.equal(S.saving,false);
succeed=true;await mark('reject','cut');assert.equal(S.i,2);assert.equal(S.clips[2].dec,'reject');
assert.equal(dialog.open,false);assert.match(document.getElementById('badge').textContent,/REJECT/);
console.log('Reject dialog: PASS');
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        program = 'new Function(' + json.dumps(script) + ');\n' + harness + script + checks
        result = subprocess.run(['node'], input=program, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
