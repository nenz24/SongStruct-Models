const LABELS=['intro','verse','pre-chorus','chorus','bridge','inst','outro','silence'];
const OWNER_DEFAULT='';
const stamp=()=>new Date().toISOString();
class APIError extends Error {constructor(status,message){super(message);this.status=status;}}
const fail=(s,m)=>{throw new APIError(s,m)};
const json=(data,status=200,extra={})=>new Response(JSON.stringify(data),{status,headers:{'content-type':'application/json; charset=utf-8','cache-control':'no-store','x-content-type-options':'nosniff',...extra}});
const query=(db,sql,...args)=>db.prepare(sql).bind(...args);
const digest=async value=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value)))).map(v=>v.toString(16).padStart(2,'0')).join('');

export function validateDocument(doc){
  if (!doc || doc.label_source!=='ASLP-lab/SongFormer' || !Array.isArray(doc.sections) || !doc.sections.length || doc.sections.length>3000) fail(400,'Pilih JSON hasil SongFormer dari outputs.local/labeled.');
  if(typeof doc.duration!=='number'||!Number.isFinite(doc.duration)||doc.duration<=0) fail(400,'Durasi lagu tidak valid.');
  let previous=0;
  for(const s of doc.sections){
    if(typeof s.start!=='number'||typeof s.end!=='number'||!Number.isFinite(s.start+s.end)||s.start<previous-.001||s.start<0||s.end<=s.start||s.end>doc.duration+.25||!LABELS.includes(s.label_initial??s.label)) fail(400,'Boundary atau label awal tidak valid.');
    if(s.label_manual!=null&&!LABELS.includes(s.label_manual)) fail(400,'Label manual tidak valid.');
    previous=s.end;
  }
  // Do not upload local machine paths or bulky raw Apple results to team clients.
  return {schema_version:doc.schema_version??1,track_id:String(doc.track_id??'').slice(0,1000),duration:doc.duration,
    boundary_source:doc.boundary_source??'apple_musicunderstanding.sections',label_source:doc.label_source,
    encoder_window_seconds:doc.encoder_window_seconds??null,songformer_revision:doc.songformer_revision??null,
    apple_json_sha256:doc.apple_json_sha256??null,
    sections:doc.sections.map((s,i)=>({id:String(s.id??`s${i+1}`).slice(0,80),start:s.start,end:s.end,
      label_initial:s.label_initial??s.label,label_scores:s.label_scores??{},needs_review:!!s.needs_review,
      coverage_fraction:s.coverage_fraction??null,label_manual:s.label_manual??null}))};
}
export function cleanEdits(edits,count){
  if(!Array.isArray(edits)||edits.length!==count) fail(400,'Jumlah section tidak sesuai.');
  return edits.map(e=>{
    if(!e||e.label_manual!==null&&!LABELS.includes(e.label_manual))fail(400,'Label manual tidak valid.');
    return {label_manual:e.label_manual,note:String(e.note??'').slice(0,2000)};
  });
}
function track(row){
  const original=JSON.parse(row.original),edits=JSON.parse(row.edits);
  return {id:row.id,title:row.title,duration:original.duration,status:row.status,assignee:row.assignee,version:row.version,
    audio_ready:!!row.audio_key,updated:row.updated,sections:original.sections.map((s,i)=>({...s,...edits[i]}))};
}
async function user(request,env){
  const OWNER=(env.OWNER_EMAIL??OWNER_DEFAULT).toLowerCase();
  const id=request.headers.get('oai-authenticated-user-id'),email=(request.headers.get('oai-authenticated-user-email')??'').toLowerCase();
  // On a portable public deployment, never trust client-supplied identity headers.
  // Hosted Sites retains platform identity only when code mode is disabled.
  if(env.INVITE_AUTH==='1'){
    const token=(request.headers.get('authorization')??'').replace(/^Bearer /,'');
    if(token.length<20||token.length>256)fail(401,'Masukkan kode akses dari pengelola.');
    const hash=await digest(token);
    if(env.ADMIN_ACCESS_CODE&&hash===await digest(env.ADMIN_ACCESS_CODE)){
      await query(env.DB,"INSERT OR IGNORE INTO users(id,email,name,role) VALUES('owner','owner@local','Pengelola','admin')").run();
      return {id:'owner',email:'owner@local',name:'Pengelola',role:'admin',auth_mode:'code'};
    }
    const member=await query(env.DB,'SELECT u.* FROM access_codes c JOIN users u ON u.id=c.user_id WHERE c.hash=?',hash).first();
    if(!member)fail(401,'Kode akses tidak valid atau sudah dicabut.');
    return {...member,auth_mode:'code'};
  }
  if(!id||!email)fail(401,'Masuk dengan akun ChatGPT untuk melanjutkan.');
  const existing=await query(env.DB,'SELECT * FROM users WHERE id=?',id).first();
  if(existing)return existing;
  const invite=await query(env.DB,'SELECT * FROM invites WHERE email=?',email).first();
  if(email!==OWNER&&!invite)fail(403,'Akun Anda belum diundang ke tim anotasi. Hubungi pengelola.');
  const raw=request.headers.get('oai-authenticated-user-full-name');
  let display=raw??email;
  if(raw&&request.headers.get('oai-authenticated-user-full-name-encoding')==='percent-encoded-utf-8'){try{display=decodeURIComponent(raw)}catch{display=email}}
  const u={id,email,name:invite?.name||display,role:email===OWNER?'admin':'annotator'};
  await query(env.DB,'INSERT OR IGNORE INTO users(id,email,name,role) VALUES(?,?,?,?)',u.id,u.email,u.name,u.role).run();
  return u;
}
const admin=u=>{if(u.role!=='admin')fail(403,'Hanya pengelola yang dapat melakukan ini.')};
async function getRow(db,id){const r=await query(db,'SELECT * FROM tracks WHERE id=?',id).first();if(!r)fail(404,'Lagu tidak ditemukan.');return r;}
async function mutation(db,row,u,action,values){
  const statements=[query(db,'UPDATE tracks SET edits=?,status=?,assignee=?,version=version+1,updated=? WHERE id=? AND version=?',
    JSON.stringify(values.edits),values.status,values.assignee,stamp(),row.id,row.version),
    query(db,'INSERT INTO events(id,track_id,user_id,action,at,details) SELECT ?,?,?,?,?,? WHERE changes()=1',
      crypto.randomUUID(),row.id,u.id,action,stamp(),JSON.stringify(values))];
  const result=await db.batch(statements);
  if(!result[0].meta.changes)fail(409,'Lagu diubah anotator lain. Muat ulang sebelum menyimpan.');
  return track(await getRow(db,row.id));
}

async function handler(request,env){
  const url=new URL(request.url),path=url.pathname;
  if(request.method==='GET'&&['/','/style.css','/app.js'].includes(path)){
    const asset=ASSETS[path==='/'?'html':path==='/style.css'?'css':'js'];
    return new Response(asset.body,{headers:{'content-type':asset.type,'cache-control':'no-cache',
      'content-security-policy':"default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
      'x-content-type-options':'nosniff','referrer-policy':'no-referrer'}});
  }
  if(!path.startsWith('/api/'))fail(404,'Tidak ditemukan.');
  if(path==='/api/config'&&request.method==='GET')return json({auth_mode:env.INVITE_AUTH==='1'?'code':'chatgpt'});
  if(!env.DB||!env.BUCKET)fail(503,'Penyimpanan cloud belum siap. Hubungi pengelola.');
  const u=await user(request,env);
  if(request.method==='GET'){
    if(path==='/api/me')return json(u);
    if(path==='/api/tracks'){
      const [rows,users,invites]=await Promise.all([env.DB.prepare('SELECT * FROM tracks ORDER BY title').all(),
        env.DB.prepare('SELECT id,name,email,role FROM users ORDER BY name').all(),
        u.role==='admin'?env.DB.prepare('SELECT email,name FROM invites ORDER BY name').all():Promise.resolve({results:[]})]);
      return json({labels:LABELS,users:users.results,invites:invites.results,tracks:rows.results.map(r=>{
        const t=track(r); t.total=t.sections.length;t.checked=t.sections.filter(s=>s.label_manual!==null).length;delete t.sections;return t;
      })});
    }
    if(path==='/api/export'){
      admin(u);
      const all=url.searchParams.get('scope')==='all';
      const rows=(await env.DB.prepare(all?'SELECT * FROM tracks ORDER BY title':"SELECT * FROM tracks WHERE status='approved' ORDER BY title").all()).results;
      const documents=rows.map(r=>{
        const doc=JSON.parse(r.original),edits=JSON.parse(r.edits);
        return {...doc,title:r.title,boundary_locked:true,review_status:r.status,annotator_id:r.assignee,
          annotation_updated:r.updated,annotation_version:r.version,
          sections:doc.sections.map((s,i)=>({...s,...edits[i],label:edits[i].label_manual??s.label_initial,review_status:r.status}))};
      });
      return json({schema_version:1,exported_at:stamp(),count:documents.length,tracks:documents},200,
        {'content-disposition':'attachment; filename="songstruct-'+(all?'all':'approved')+'.json"'});
    }
    const match=path.match(/^\/api\/tracks\/([a-f0-9-]+)(\/audio)?$/);
    if(match){
      const row=await getRow(env.DB,match[1]);
      if(!match[2])return json(track(row));
      if(!row.audio_key)fail(404,'Audio belum diunggah oleh pengelola.');
      const object=await env.BUCKET.get(row.audio_key, {range:request.headers});
      if(!object)fail(404,'Audio tidak ditemukan. Unggah kembali.');
      const headers=new Headers({'content-type':'audio/mpeg','accept-ranges':'bytes','cache-control':'private, max-age=1800','x-content-type-options':'nosniff'});
      const range=object.range;
      if(range?.length!==undefined&&range?.offset!==undefined){
        headers.set('content-range',`bytes ${range.offset}-${range.offset+range.length-1}/${object.size}`);
        headers.set('content-length',String(range.length));
      }else headers.set('content-length',String(object.size));
      return new Response(object.body,{status:range?206:200,headers});
    }
  }
  if(!['POST','PUT'].includes(request.method))fail(405,'Metode tidak didukung.');
  // Same-origin requests only. Access and roles are checked on every API call.
  if(request.headers.get('x-requested-with')!=='SongStruct'||request.headers.get('origin')&&request.headers.get('origin')!==url.origin)fail(403,'Permintaan harus berasal dari aplikasi ini.');
  const audioMatch=path.match(/^\/api\/tracks\/([a-f0-9-]+)\/audio$/);
  if(audioMatch&&request.method==='PUT'){
    admin(u);const row=await getRow(env.DB,audioMatch[1]);
    const length=Number(request.headers.get('content-length'));
    if(!Number.isFinite(length)||length<=0||length>90*1024*1024)fail(413,'Audio maksimal 90 MB per file.');
    const key='audio/'+row.id+'/'+crypto.randomUUID()+'.mp3';
    await env.BUCKET.put(key,request.body,{httpMetadata:{contentType:'audio/mpeg'}});
    await query(env.DB,'UPDATE tracks SET audio_key=?,audio_size=?,updated=? WHERE id=?',key,length,stamp(),row.id).run();
    if(row.audio_key)await env.BUCKET.delete(row.audio_key);
    return json({ok:true});
  }
  const text=await request.text();
  if(text.length>2_000_000)fail(413,'JSON terlalu besar.');
  let data;try{data=JSON.parse(text)}catch{fail(400,'JSON tidak valid.')}
  if(path==='/api/users'){
    admin(u);const email=String(data.email??'').trim().toLowerCase(),name=String(data.name??'').trim().slice(0,80);
    if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)||!name)fail(400,'Isi nama dan email yang valid.');
    await query(env.DB,'INSERT INTO invites(email,name) VALUES(?,?) ON CONFLICT(email) DO UPDATE SET name=excluded.name',email,name).run();
    return json({ok:true,note:'Anggota terdaftar di aplikasi. Akses situs juga perlu dibagikan ke email yang sama melalui menu berbagi Sites.'});
  }
  if(path==='/api/codes'){
    admin(u);if(env.INVITE_AUTH!=='1')fail(400,'Kode akses hanya tersedia pada deployment Cloudflare mandiri.');
    const name=String(data.name??'').trim().slice(0,80);if(!name)fail(400,'Isi nama anotator.');
    const id=crypto.randomUUID(),code=crypto.randomUUID()+crypto.randomUUID().replaceAll('-','');
    await env.DB.batch([query(env.DB,'INSERT INTO users(id,email,name,role) VALUES(?,?,?,?)',id,id+'@code.local',name,'annotator'),
      query(env.DB,'INSERT INTO access_codes(hash,user_id) VALUES(?,?)',await digest(code),id)]);
    return json({id,name,code});
  }
  if(path==='/api/revoke'){
    admin(u);if(env.INVITE_AUTH!=='1')fail(400,'Tidak tersedia pada mode Sites.');
    if(!data.id||data.id==='owner')fail(400,'Anggota tidak valid.');
    await query(env.DB,'DELETE FROM access_codes WHERE user_id=?',data.id).run();return json({ok:true});
  }
  if(path==='/api/import'){
    admin(u);const doc=validateDocument(data.document);
    const title=String(data.title??doc.track_id.split('/').pop()??'Lagu').slice(0,300);
    const id=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(doc.track_id||title)))).map(v=>v.toString(16).padStart(2,'0')).join('').slice(0,24);
    const exists=await query(env.DB,'SELECT id FROM tracks WHERE id=?',id).first();
    if(exists)return json({id,exists:true});
    const edits=doc.sections.map(s=>({label_manual:s.label_manual,note:''}));
    await query(env.DB,'INSERT INTO tracks(id,title,original,edits,status,created,updated) VALUES(?,?,?,?,?,?,?)',id,title,JSON.stringify(doc),JSON.stringify(edits),'pending',stamp(),stamp()).run();
    return json({id,exists:false});
  }
  const match=path.match(/^\/api\/tracks\/([a-f0-9-]+)\/(claim|assign|save|approve|reopen)$/);
  if(!match)fail(404,'Tidak ditemukan.');
  const row=await getRow(env.DB,match[1]),action=match[2];
  if(data.version!==row.version)fail(409,'Lagu diubah anotator lain. Muat ulang lagu sebelum menyimpan.');
  let edits=JSON.parse(row.edits),status=row.status,assignee=row.assignee;
  if(action==='claim'){
    if(assignee&&assignee!==u.id)fail(409,'Lagu sudah diambil anotator lain.');
    if(['submitted','approved'].includes(status))fail(409,'Lagu sudah dikirim.');
    assignee=u.id;status='in_progress';
  }else if(action==='assign'){
    admin(u);assignee=data.assignee||null;
    if(assignee&&!await query(env.DB,'SELECT id FROM users WHERE id=?',assignee).first())fail(400,'Anotator tidak ditemukan.');
    if(['submitted','approved'].includes(status))fail(409,'Buka kembali sebelum mengubah penugasan.');
    status=assignee?'in_progress':'pending';
  }else if(action==='save'){
    if(u.role!=='admin'&&assignee!==u.id)fail(403,'Ambil tugas ini sebelum mengedit.');
    if(['submitted','approved'].includes(status))fail(409,'Pengelola perlu membuka kembali anotasi ini.');
    edits=cleanEdits(data.edits,JSON.parse(row.original).sections.length);
    if(data.submit&&edits.some(e=>e.label_manual===null))fail(400,'Periksa seluruh section sebelum mengirim.');
    status=data.submit?'submitted':'in_progress';assignee=assignee||u.id;
  }else if(action==='approve'){
    admin(u);if(status!=='submitted'||edits.some(e=>e.label_manual===null))fail(400,'Anotasi harus lengkap dan sudah dikirim.');status='approved';
  }else if(action==='reopen'){
    admin(u);status=assignee?'in_progress':'pending';
  }
  return json(await mutation(env.DB,row,u,action,{edits,status,assignee}));
}

export default {async fetch(request,env){try{return await handler(request,env)}catch(error){
  if(!(error instanceof APIError))console.error(error);
  return json({error:error instanceof APIError?error.message:'Terjadi kesalahan server. Hasil belum dikonfirmasi tersimpan.'},error.status??500);
}}};
