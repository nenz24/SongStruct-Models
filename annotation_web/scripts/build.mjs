import {readFile,mkdir,writeFile,cp} from 'node:fs/promises';
const assets = {};
for (const [name,file,type] of [['html','index.html','text/html; charset=utf-8'],['css','style.css','text/css; charset=utf-8'],['js','app.js','text/javascript; charset=utf-8']]) assets[name] = {body:await readFile('ui/'+file,'utf8'),type};
await mkdir('dist/server',{recursive:true}); await mkdir('dist/.openai',{recursive:true});
const code=await readFile('worker/index.js','utf8');
await writeFile('dist/server/index.js','const ASSETS='+JSON.stringify(assets)+';\n'+code);
try{await cp('.openai/hosting.json','dist/.openai/hosting.json')}catch(e){if(e.code!=='ENOENT')throw e};
await cp('drizzle','dist/drizzle',{recursive:true});
console.log('Built cloud Worker, UI assets, and schema migrations.');
