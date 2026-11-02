// Exercise the real client controller with controlled asynchronous transport.
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
function setup(){
 const elements=new Map(),events=new Map();
 function el(id){if(!elements.has(id))elements.set(id,{id,hidden:true,disabled:false,value:id==='severity'?'1':id==='corruption'?'none':'',innerHTML:'',textContent:'',classList:{add(){},remove(){},toggle(){},contains(){return false}},addEventListener(){},removeAttribute(){},getBoundingClientRect(){return{width:0,height:0}},click(){},scrollIntoView(){}});return elements.get(id);}
 const context=vm.createContext({console,URL:{createObjectURL(){return'blob:test'},revokeObjectURL(){}},document:{getElementById:el,querySelector:el,querySelectorAll(){return[]},body:{classList:{toggle(){},remove(){}}}},window:{addEventListener(name,handler){events.set(name,handler)},scrollTo(){}},location:{hash:'#diagnose'},matchMedia:()=>({matches:true}),setTimeout:()=>1,clearTimeout(){},requestAnimationFrame(){},FormData:class{append(){}},AbortSignal:{timeout(){}},performance:{now:()=>0},fetch:async()=>({ok:true,json:async()=>[]})});
 let source=fs.readFileSync(require('node:path').join(__dirname,'../demo/static/app.js'),'utf8');
 source=source.replace('navigate();refreshStatus();loadExamples();drawPreview();requestAnimationFrame(animate);','');
 vm.runInContext(source,context);
 vm.runInContext("state.status={models:{baseline:true,de:true}};pickFile({name:'first.png',type:'image/png',size:128});",context);
 return{context,el,events,run:code=>vm.runInContext(code,context)};
}
for(const inputChange of ['new image','corruption','severity'])test(`stale response is ignored after ${inputChange}`,async()=>{
 const app=setup();let resolve;
 app.context.fetch=()=>new Promise(r=>resolve=r);
 const ongoing=app.el('analyze-button').onclick();
 if(inputChange==='new image')app.run("pickFile({name:'second.png',type:'image/png',size:128})");
 else if(inputChange==='corruption'){app.el('corruption').value='blur';app.el('corruption').onchange();}
 else{app.el('severity').value='2';app.el('severity').onchange();}
 resolve({ok:true,json:async()=>({models:[]})});await ongoing;
 assert.equal(app.el('prediction-results').hidden,true);
 assert.equal(app.el('prediction-empty').hidden,false);
 assert.equal(app.el('analyze-button').disabled,false);
});
test('current prediction is displayed',async()=>{const app=setup();app.context.fetch=async()=>({ok:true,json:async()=>({models:[]})});await app.el('analyze-button').onclick();assert.equal(app.el('prediction-results').hidden,false);assert.equal(app.el('prediction-loading').hidden,true);});
test('clearing an image invalidates an active prediction',async()=>{const app=setup();let resolve;app.context.fetch=()=>new Promise(r=>resolve=r);const ongoing=app.el('analyze-button').onclick();app.el('clear-image').onclick();resolve({ok:true,json:async()=>({models:[]})});await ongoing;assert.equal(app.el('prediction-results').hidden,true);assert.equal(app.el('analyze-button').disabled,true);});
test('invalid uploads do not replace valid selection',()=>{const app=setup();app.run("pickFile({name:'bad.txt',type:'text/plain',size:128})");assert.equal(app.el('file-label').textContent,'first.png');});

test('Space preserves native activation on focused controls',()=>{
 const app=setup();app.context.location.hash='#evolution';app.run('state.experiment={};');let toggles=0,prevented=false;
 app.context.togglePlayback=()=>toggles++;
 app.events.get('keydown')({key:' ',code:'Space',target:{closest:()=>({tagName:'BUTTON'})},preventDefault(){prevented=true;}});
 assert.equal(toggles,0);assert.equal(prevented,false);
});
test('Space still controls playback outside interactive controls',()=>{
 const app=setup();app.context.location.hash='#evolution';app.run('state.experiment={};');let toggles=0,prevented=false;
 app.context.togglePlayback=()=>toggles++;
 app.events.get('keydown')({key:' ',code:'Space',target:{closest:()=>null},preventDefault(){prevented=true;}});
 assert.equal(toggles,1);assert.equal(prevented,true);
});
test('presentation shortcut still works after focusing a button',()=>{
 const app=setup();let toggles=0;app.el('present-button').click=()=>toggles++;
 app.events.get('keydown')({key:'p',code:'KeyP',target:{closest:selector=>selector.includes('button')?{}:null},preventDefault(){}});
 assert.equal(toggles,1);
});
test('browser and operating-system shortcuts are not treated as presentation keys',()=>{
 const app=setup();let toggles=0;app.el('present-button').click=()=>toggles++;
 for(const modifier of ['ctrlKey','metaKey','altKey'])app.events.get('keydown')({key:'p',code:'KeyP',[modifier]:true,target:{closest:()=>null},preventDefault(){}});
 assert.equal(toggles,0);
});
test('Escape redraws charts after the presentation workspace changes width',()=>{
 const app=setup();let draws=0;
 app.context.requestAnimationFrame=callback=>callback();app.context.redrawPlots=()=>draws++;
 app.events.get('keydown')({key:'Escape',code:'Escape',target:{closest:()=>null},preventDefault(){}});
 assert.equal(draws,1);
});

// Synthetic fixtures exercise presentation logic; never used as experiment data.
function resultFixture(){
 const perSeed=(f1)=>Array.from({length:3},()=>({per_class:[{f1}],calibration:{ece:.025},bootstrap_95_ci:{accuracy:[.85,.95]}}));
 const stress={results:Object.fromEntries(['blur','noise','brightness','jpeg','low_light'].map(k=>[k,{'1':{accuracy:.9},'2':{accuracy:.8},'3':{accuracy:.7}}]))};
 return{class_names:['Tomato___Synthetic_<leaf>'],models:{baseline:{per_seed:perSeed(.9)},de:{per_seed:perSeed(.7)}},robustness:{baseline:stress,de:stress}};
}
test('result view preserves negative class gains and escapes labels',()=>{const app=setup();app.context.fixture=resultFixture();const html=app.run('extraResults(fixture)');assert.match(html,/negative-gain">-20\.0/);assert.match(html,/Synthetic &lt;leaf&gt;/);assert.doesNotMatch(html,/Synthetic <leaf>/);});
test('result view shows mean corruption performance and seed-zero intervals',()=>{const app=setup();app.context.fixture=resultFixture();const html=app.run('extraResults(fixture)');assert.match(html,/80\.0%/);assert.match(html,/85\.0% – 95\.0%/);assert.match(html,/2\.50%/);});
test('headline interpretation preserves a tied control and worse robustness',()=>{
 const app=setup();app.context.fixture={models:{baseline:{aggregate:{accuracy:{mean:.97}}},de:{aggregate:{accuracy:{mean:.99}}},rs:{aggregate:{accuracy:{mean:.99}}}},robustness:{baseline:{mean_accuracy:.86},de:{mean_accuracy:.69}}};
 const html=app.run('evidenceTakeaways(fixture)');assert.match(html,/Same mean accuracy/);assert.match(html,/does not show an accuracy advantage/);assert.match(html,/Lower stress-test accuracy/);assert.match(html,/DE 69\.00% · baseline 86\.00%/);
});
test('headline interpretation reports negative clean gains and does not round a control gap into a tie',()=>{
 const app=setup();app.context.fixture={models:{baseline:{aggregate:{accuracy:{mean:.99}}},de:{aggregate:{accuracy:{mean:.97001}}},rs:{aggregate:{accuracy:{mean:.97002}}}},robustness:{baseline:{mean_accuracy:.69},de:{mean_accuracy:.86}}};
 const html=app.run('evidenceTakeaways(fixture)');assert.match(html,/-2\.00 percentage points/);assert.match(html,/DE behind random search/);assert.doesNotMatch(html,/Same mean accuracy/);assert.match(html,/Higher or equal stress-test accuracy/);
});
test('reopening diagnosis loads examples that became available after startup',async()=>{
 const app=setup();let galleryCalls=0;
 app.context.fetch=async url=>({ok:true,json:async()=>url==='/api/status'?{models:{baseline:true,de:true},class_count:10,data:{usable_total:20}}:++galleryCalls===1?[]:[{file:'sample.jpg',class_name:'Tomato___healthy'}]});
 await app.run('loadExamples()');assert.equal(app.el('example-gallery').innerHTML,'');
 app.run('navigate()');await new Promise(resolve=>setImmediate(resolve));
 assert.equal(galleryCalls,2);assert.match(app.el('example-gallery').innerHTML,/sample.jpg/);
});
