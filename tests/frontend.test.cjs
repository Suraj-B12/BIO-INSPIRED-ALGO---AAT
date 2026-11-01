// Exercise the real client controller with controlled asynchronous transport.
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
function setup(){
 const elements=new Map();
 function el(id){if(!elements.has(id))elements.set(id,{id,hidden:true,disabled:false,value:id==='severity'?'1':id==='corruption'?'none':'',innerHTML:'',textContent:'',classList:{add(){},remove(){},toggle(){},contains(){return false}},addEventListener(){},removeAttribute(){},getBoundingClientRect(){return{width:0,height:0}},click(){},scrollIntoView(){}});return elements.get(id);}
 const context=vm.createContext({console,URL:{createObjectURL(){return'blob:test'},revokeObjectURL(){}},document:{getElementById:el,querySelector:el,querySelectorAll(){return[]},body:{classList:{toggle(){},remove(){}}}},window:{addEventListener(){},scrollTo(){}},location:{hash:'#diagnose'},matchMedia:()=>({matches:true}),setTimeout:()=>1,clearTimeout(){},requestAnimationFrame(){},FormData:class{append(){}},AbortSignal:{timeout(){}},performance:{now:()=>0},fetch:async()=>({ok:true,json:async()=>[]})});
 let source=fs.readFileSync(require('node:path').join(__dirname,'../demo/static/app.js'),'utf8');
 source=source.replace('navigate();refreshStatus();loadExamples();drawPreview();requestAnimationFrame(animate);','');
 vm.runInContext(source,context);
 vm.runInContext("state.status={models:{baseline:true,de:true}};pickFile({name:'first.png',type:'image/png',size:128});",context);
 return{context,el,run:code=>vm.runInContext(code,context)};
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
