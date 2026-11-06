import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import worker,{validateDocument} from '../worker/index.js';

function database(){
 const db=new DatabaseSync(':memory:');db.exec(readFileSync('drizzle/0000_annotation.sql','utf8'));db.exec(readFileSync('drizzle/0001_access_codes.sql','utf8'));db.exec(readFileSync('drizzle/0002_blocked_links.sql','utf8'));
 const api={prepare(sql){let args=[];const obj={bind(...v){args=v;return obj},async first(){return db.prepare(sql).get(...args)??null},async all(){return {results:db.prepare(sql).all(...args)}},async run(){const r=db.prepare(sql).run(...args);return {meta:{changes:Number(r.changes)}}}};return obj},async batch(statements){db.exec('BEGIN');try{const results=[];for(const s of statements)results.push(await s.run());db.exec('COMMIT');return results}catch(e){db.exec('ROLLBACK');throw e}}};
 return api;
}
const doc={track_id:'folder/song.mp3',audio_path:'/private/path/song.mp3',duration:20,label_source:'ASLP-lab/SongFormer',sections:[{start:0,end:10,label_initial:'verse'},{start:10,end:20,label_initial:'chorus'}]};
function setup(){const values=new Map();const env={OWNER_EMAIL:'owner@example.com',DB:database(),BUCKET:{async put(k,body){values.set(k,new Uint8Array(await new Response(body).arrayBuffer()))},async delete(k){values.delete(k)},async get(k,opts){const data=values.get(k);if(!data)return null;const raw=opts?.range?.get('range');let range,body=data;if(raw){const m=raw.match(/bytes=(\d+)-(\d*)/);const start=Number(m[1]),end=m[2]?Number(m[2]):data.length-1;body=data.slice(start,end+1);range={offset:start,length:body.length}}return {body,size:data.length,range}}}};
 async function req(path,{method='GET',body,user='owner',headers={}}={}){
  const h={'oai-authenticated-user-id':user,'oai-authenticated-user-email':user==='owner'?'owner@example.com':user+'@example.com',...headers};
  if(body!==undefined){h['x-requested-with']='SongStruct';h.origin='https://studio.test';h['content-type']='application/json';body=JSON.stringify(body)}
  return worker.fetch(new Request('https://studio.test/api'+path,{method,headers:h,body}),env);
 }return {req,env};}

test('validation strips local path and preserves Apple boundaries',()=>{
 const clean=validateDocument(doc);assert.equal(clean.audio_path,undefined);assert.deepEqual(clean.sections.map(s=>[s.start,s.end]),[[0,10],[10,20]]);
 assert.throws(()=>validateDocument({...doc,sections:[{start:0,end:30,label:'verse'}]}));
});
test('collaboration, claim conflict, locked boundaries, review, export and media ranges',async()=>{
 const {req,env}=setup();
 assert.equal((await req('/tracks',{user:'stranger'})).status,403);
 const importRes=await req('/import',{method:'POST',body:{document:doc,title:'Worship pilot'}});assert.equal(importRes.status,200);const {id}=await importRes.json();
 assert.equal((await req('/import',{method:'POST',body:{document:doc}})).status,200);
 await req('/users',{method:'POST',body:{name:'Alice',email:'alice@example.com'}});
 await req('/users',{method:'POST',body:{name:'Bob',email:'bob@example.com'}});
 await req('/me',{user:'alice'});
 let r=await req('/tracks/'+id+'/assign',{method:'POST',body:{version:0,assignee:'alice'}});assert.equal(r.status,200);let track=await r.json();
 assert.equal((await req('/tracks/'+id+'/claim',{method:'POST',user:'bob',body:{version:0}})).status,404);
 assert.equal((await req('/tracks/'+id+'/save',{method:'POST',user:'bob',body:{version:1,edits:[]}})).status,404);
 const edits=[{label_manual:'verse',note:'Sudah benar'},{label_manual:'bridge',note:'Koreksi chorus ke bridge'}];
 r=await req('/tracks/'+id+'/save',{method:'POST',user:'alice',body:{version:1,edits,submit:true}});assert.equal(r.status,200);track=await r.json();assert.equal(track.status,'submitted');
 assert.equal((await req('/tracks/'+id+'/save',{method:'POST',user:'alice',body:{version:2,edits}})).status,409);
 assert.equal((await req('/tracks/'+id+'/approve',{method:'POST',user:'alice',body:{version:2}})).status,403);
 r=await req('/tracks/'+id+'/approve',{method:'POST',body:{version:2}});assert.equal(r.status,200);
 const exported=await (await req('/export')).json();assert.equal(exported.count,1);assert.equal(exported.tracks[0].sections[1].label,'bridge');assert.deepEqual(exported.tracks[0].sections.map(s=>[s.start,s.end]),[[0,10],[10,20]]);
 assert.equal(exported.tracks[0].sections[1].label_initial,'chorus');
 assert.equal((await req('/export',{user:'alice'})).status,403);
 const media=new Uint8Array([1,2,3,4,5]);
 let audioRes=await worker.fetch(new Request('https://studio.test/api/tracks/'+id+'/audio',{method:'PUT',headers:{'oai-authenticated-user-id':'owner','oai-authenticated-user-email':'owner@example.com','x-requested-with':'SongStruct','origin':'https://studio.test','content-length':'5'},body:media}),env);
 assert.equal(audioRes.status,200);
 audioRes=await req('/tracks/'+id+'/audio',{user:'alice',headers:{range:'bytes=1-3'}});assert.equal(audioRes.status,206);assert.deepEqual([...new Uint8Array(await audioRes.arrayBuffer())],[2,3,4]);
 assert.equal(audioRes.headers.get('content-range'),'bytes 1-3/5');
});
test('same-origin mutation protection',async()=>{
 const {env}=setup();const res=await worker.fetch(new Request('https://studio.test/api/import',{method:'POST',headers:{'oai-authenticated-user-id':'owner','oai-authenticated-user-email':'owner@example.com','x-requested-with':'SongStruct',origin:'https://other.test'},body:JSON.stringify({document:doc})}),env);assert.equal(res.status,403);
});

test('portable code access, per-person attribution, revocation and export protection',async()=>{
 const {env}=setup();env.INVITE_AUTH='1';env.ADMIN_ACCESS_CODE='a-very-long-random-admin-code-for-test';
 async function call(path,token,body){return worker.fetch(new Request('https://studio.test/api'+path,{method:body?'POST':'GET',headers:{authorization:'Bearer '+token,'x-requested-with':'SongStruct',origin:'https://studio.test'},body:body?JSON.stringify(body):undefined}),env)}
 assert.equal((await call('/me','incorrect-token-should-be-denied')).status,401);
 const spoofed=await worker.fetch(new Request('https://studio.test/api/me',{headers:{'oai-authenticated-user-id':'owner','oai-authenticated-user-email':'forged@example.com'}}),env);assert.equal(spoofed.status,401);
 const owner=await (await call('/me',env.ADMIN_ACCESS_CODE)).json();assert.equal(owner.role,'admin');
 const invite=await (await call('/codes',env.ADMIN_ACCESS_CODE,{name:'Anotator 1'})).json();assert.ok(invite.code.length>30);
 const person=await (await call('/me',invite.code)).json();assert.equal(person.name,'Anotator 1');assert.equal(person.role,'annotator');
 assert.equal((await call('/export',invite.code)).status,403);
 await call('/revoke',env.ADMIN_ACCESS_CODE,{id:person.id});assert.equal((await call('/me',invite.code)).status,401);
});

test('managed Juan/Jordan/Otniel links, balanced allocation, uploader scope and durable revocation',async()=>{
 const {env}=setup();env.INVITE_AUTH='1';env.ADMIN_ACCESS_CODE='a-long-admin-code-for-managed-team';env.TEAM_LINK_KEY='test-secret-at-least-32-characters-long';
 async function call(path,token,body,method){return worker.fetch(new Request('https://studio.test/api'+path,{method:method||(body?'POST':'GET'),headers:{...(token?{authorization:'Bearer '+token}:{}),'x-requested-with':'SongStruct',origin:'https://studio.test'},body:body?JSON.stringify(body):undefined}),env)}
 assert.equal((await call('/config')).status,200);
 const sharing=await (await call('/share-links',env.ADMIN_ACCESS_CODE)).json();assert.deepEqual(sharing.links.map(m=>m.name),['Juan','Jordan','Otniel']);
 const upload=sharing.upload_config.upload_code;assert.equal((await call('/export',upload)).status,403);assert.equal((await call('/tracks',upload)).status,403);
 for(let i=0;i<7;i++)assert.equal((await call('/import',upload,{document:{...doc,track_id:'song'+i+'.mp3'},title:'Song '+i})).status,200);
 const tracks=(await (await call('/tracks',env.ADMIN_ACCESS_CODE)).json()).tracks;
 assert.deepEqual(['team_juan','team_jordan','team_otniel'].map(id=>tracks.filter(t=>t.assignee===id).length),[3,2,2]);
 const one=tracks[0];assert.equal((await call('/tracks/'+one.id+'/save',upload,{version:one.version,edits:[]})).status,403);
 const juanToken=new URL(sharing.links[0].url).hash.slice('#access='.length);
 assert.equal((await (await call('/me',juanToken)).json()).name,'Juan');
 const mine=await (await call('/tracks',juanToken)).json();assert.equal(mine.tracks.length,3);assert.equal(mine.users.length,1);assert.ok(mine.tracks.every(t=>t.assignee==='team_juan'));
 const other=tracks.find(t=>t.assignee==='team_jordan');
 assert.equal((await call('/tracks/'+other.id,juanToken)).status,404);
 assert.equal((await call('/tracks/'+other.id+'/audio',juanToken)).status,404);
 assert.equal((await call('/tracks/'+other.id+'/save',juanToken,{version:other.version,edits:[]})).status,404);
 assert.equal((await call('/tracks/'+other.id+'/claim',juanToken,{version:other.version})).status,404);
 const own=mine.tracks[0];assert.equal((await call('/tracks/'+own.id,juanToken)).status,200);
 assert.equal((await call('/tracks/'+other.id,env.ADMIN_ACCESS_CODE)).status,200);

 await call('/revoke',env.ADMIN_ACCESS_CODE,{id:'team_juan'});await call('/config');assert.equal((await call('/me',juanToken)).status,401);
 env.HOSTED_SITES='1';
 const owner=await worker.fetch(new Request('https://studio.test/api/me',{headers:{'oai-authenticated-user-id':'owner-id','oai-authenticated-user-email':'owner@example.com'}}),env);assert.equal((await owner.json()).role,'admin');
});
