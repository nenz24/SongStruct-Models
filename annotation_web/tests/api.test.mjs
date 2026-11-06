import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import worker,{validateDocument} from '../worker/index.js';

function database(){
 const db=new DatabaseSync(':memory:');db.exec(readFileSync('drizzle/0000_annotation.sql','utf8'));db.exec(readFileSync('drizzle/0001_access_codes.sql','utf8'));
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
 let r=await req('/tracks/'+id+'/claim',{method:'POST',user:'alice',body:{version:0}});assert.equal(r.status,200);let track=await r.json();
 assert.equal((await req('/tracks/'+id+'/claim',{method:'POST',user:'bob',body:{version:0}})).status,409);
 assert.equal((await req('/tracks/'+id+'/save',{method:'POST',user:'bob',body:{version:1,edits:[]}})).status,403);
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
 const owner=await (await call('/me',env.ADMIN_ACCESS_CODE)).json();assert.equal(owner.role,'admin');
 const invite=await (await call('/codes',env.ADMIN_ACCESS_CODE,{name:'Anotator 1'})).json();assert.ok(invite.code.length>30);
 const person=await (await call('/me',invite.code)).json();assert.equal(person.name,'Anotator 1');assert.equal(person.role,'annotator');
 assert.equal((await call('/export',invite.code)).status,403);
 await call('/revoke',env.ADMIN_ACCESS_CODE,{id:person.id});assert.equal((await call('/me',invite.code)).status,401);
});
