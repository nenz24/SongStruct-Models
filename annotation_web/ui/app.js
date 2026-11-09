const $=s=>document.querySelector(s);
const $$=s=>[...document.querySelectorAll(s)];
const state={me:null,tracks:[],users:[],invites:[],labels:[],track:null,filter:'all',search:'',dirty:false,busy:false,section:-1,loop:false,authMode:'chatgpt',managedTeam:false,shares:[]};
const statuses={pending:'Belum diambil',in_progress:'Dikerjakan',submitted:'Menunggu review',approved:'Disetujui'};
const names={intro:'Intro',verse:'Verse','pre-chorus':'Pre-chorus',chorus:'Chorus',bridge:'Bridge',inst:'Instrumental',outro:'Outro',silence:'Silence'};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=t=>`${Math.floor((t||0)/60)}:${String(Math.floor((t||0)%60)).padStart(2,'0')}`;
const fmtFine=t=>`${fmt(t)}.${Math.floor(((t||0)%1)*10)}`;
const person=id=>state.users.find(u=>u.id===id)?.name??'Belum ditugaskan';
let audio=null,stopAt=null,noticeTimer,blobURL=null;
let accessCode=sessionStorage.getItem('songstruct-code')||'';
const invitation=new URLSearchParams(location.hash.slice(1)).get('access');
if(invitation){accessCode=invitation;sessionStorage.setItem('songstruct-code',accessCode);history.replaceState(null,'',location.pathname+location.search);}
const authHeaders=()=>accessCode?{authorization:'Bearer '+accessCode}:{};
function compactDocument(doc){
  const out={};
  for(const k of ['schema_version','track_id','duration','boundary_source','label_source','encoder_window_seconds','songformer_revision','apple_json_sha256'])if(k in doc)out[k]=doc[k];
  if(!Array.isArray(doc.sections))throw new Error('JSON tidak memiliki sections.');
  out.sections=doc.sections.map(s=>{const item={};for(const k of ['id','start','end','label_initial','label_manual','label_scores','needs_review','coverage_fraction'])if(k in s)item[k]=s[k];item.label_initial=s.label_initial??s.label;return item;});
  return out;
}
async function api(path,body,method='POST'){
  const opts=body===undefined?{headers:authHeaders()}:{method,headers:{...authHeaders(),'content-type':'application/json','x-requested-with':'SongStruct'},body:JSON.stringify(body)};
  const res=await fetch('/api'+path,opts);
  const result=await res.json();
  if(!res.ok){
    const error=new Error(result.error||'Permintaan gagal.');error.status=res.status;
    if(res.status===401){$('#auth-screen').classList.remove('hidden');$('#auth-message').textContent=result.error;}
    throw error;
  }
  return result;
}
function notice(message,error=false){
  clearTimeout(noticeTimer);const el=$('#notice');el.textContent=message;el.className='notice'+(error?' error':'');
  noticeTimer=setTimeout(()=>el.classList.add('hidden'),error?20000:8000);
}
function admin(){return state.me?.role==='admin'}
function editable(){return state.track&&!['submitted','approved'].includes(state.track.status)&&(admin()||state.track.assignee===state.me.id)}
function setDirty(){state.dirty=true;if($('#approve-all'))$('#approve-all').disabled=true;const el=$('#save-state');if(el){el.textContent='● Perubahan belum disimpan';el.classList.add('dirty')}updateChecked()}
function updateChecked(){if(!state.track)return;const count=state.track.sections.filter(s=>s.label_manual!==null).length;const el=$('#section-count');if(el)el.textContent=`${count} / ${state.track.sections.length} diperiksa`;}
function resetAudio(){if(blobURL){URL.revokeObjectURL(blobURL);blobURL=null}if(audio){audio.pause();audio.removeAttribute('src');audio.load();audio=null}stopAt=null;state.section=-1;}
async function refresh(){
  const data=await api('/tracks');state.tracks=data.tracks;state.users=data.users;state.invites=data.invites;state.labels=data.labels;
  renderList();renderTeam();renderStats();
}
function renderStats(){
  const pending=state.tracks.filter(t=>t.status==='submitted'&&t.checked===t.total);const bulk=$('#approve-all');bulk.textContent='✓ Setujui semua ('+pending.length+')';bulk.disabled=!admin()||!pending.length||state.busy||state.dirty;
  $('#review-count').textContent=state.tracks.filter(t=>t.status==='submitted').length;$('#nav-count').textContent=state.tracks.length;$('#stat-total').textContent=state.tracks.length;
  for(const key of ['in_progress','submitted','approved'])$('#stat-'+(key==='in_progress'?'progress':key)).textContent=state.tracks.filter(t=>t.status===key).length;
}
function renderList(){
  const list=state.tracks.filter(t=>t.title.toLowerCase().includes(state.search)&&(state.filter==='all'||state.filter==='mine'&&t.assignee===state.me.id||['pending','submitted','approved','in_progress'].includes(state.filter)&&t.status===state.filter));
  $('#library-count').textContent=`${list.length} lagu`;
  $('#track-list').innerHTML=list.length?list.map(t=>`<button class="track-item ${state.track?.id===t.id?'selected':''}" data-track="${esc(t.id)}"><span class="track-art">♫</span><span class="track-meta"><strong>${esc(t.title)}</strong><small>${fmt(t.duration)} &nbsp;·&nbsp; ${t.total} section${!t.audio_ready?' · audio belum ada':''}</small><span class="track-bottom"><span class="badge ${esc(t.status)}">${statuses[t.status]}</span><span>${esc(t.assignee?person(t.assignee):'Belum dibagi')} · ${t.checked}/${t.total} ✓</span></span></span></button>`).join(''):`<div class="empty small">${state.tracks.length?'Tidak ada lagu yang cocok.':admin()?'Koleksi masih kosong. Pengelola dapat mengunggah audio dan JSON berlabel.':'Belum ada lagu yang ditugaskan kepada Anda. Tugas akan muncul setelah pengelola mengunggahnya.'}</div>`;
  $$('[data-track]').forEach(b=>b.onclick=()=>openTrack(b.dataset.track));
}
async function openTrack(id){
  if(state.dirty&&!confirm('Ada perubahan belum disimpan. Tinggalkan lagu ini?'))return;
  try{const track=await api('/tracks/'+id);resetAudio();state.track=track;state.dirty=false;renderEditor();renderList()}catch(e){notice(e.message,true)}
}
function renderEditor(){
  const t=state.track;if(!t)return;
  const canEdit=editable();
  $('#editor').innerHTML=`<div class="editor-head"><div><div class="eyebrow">SEDANG DITINJAU</div><h2>${esc(t.title)}</h2><div class="editor-subline"><span class="badge ${esc(t.status)}">${statuses[t.status]}</span><span>${fmt(t.duration)} · ${t.sections.length} section</span></div></div><span class="locked">◇ Boundary terkunci</span></div>
  <div class="audio-card">${t.audio_ready?`<audio id="audio" preload="metadata" ${state.authMode==='code'?'':`src="/api/tracks/${esc(t.id)}/audio"`}></audio><div class="audio-top"><button class="play-main" id="play-main" aria-label="Putar atau jeda">▶</button><div class="audio-time"><b id="current-time">0:00</b> <span>/ ${fmt(t.duration)}</span></div><select class="rate" id="rate" aria-label="Kecepatan putar"><option value="0.75">0.75×</option><option value="1" selected>1×</option><option value="1.25">1.25×</option><option value="1.5">1.5×</option></select></div>`:`<div class="audio-placeholder">Audio belum diunggah. Pengelola perlu mengunggah MP3 yang sama dengan JSON ini.</div>`}
  <div class="timeline" aria-label="Timeline section Apple">${t.sections.map((s,i)=>`<button class="timeline-segment" data-play-section="${i}" title="${esc(names[s.label_initial])} · ${fmt(s.start)}–${fmt(s.end)}">${s.end-s.start>t.duration/18?esc(names[s.label_initial]):''}</button>`).join('')}<span class="playhead" id="playhead"></span></div><input type="range" class="seek" id="seek" min="0" max="${t.duration}" step="0.1" value="0" aria-label="Posisi pemutaran" ${!t.audio_ready?'disabled':''}><div class="timeline-labels"><span>0:00</span><span>${fmt(t.duration/2)}</span><span>${fmt(t.duration)}</span></div></div>
  <div class="assignment"><span>Anotator: <strong>${esc(person(t.assignee))}</strong></span>${admin()&&!['submitted','approved'].includes(t.status)?`<select id="assign-user" aria-label="Tugaskan ke anotator"><option value="">Belum ditugaskan</option>${state.users.map(u=>`<option value="${esc(u.id)}" ${t.assignee===u.id?'selected':''}>${esc(u.name)}</option>`).join('')}</select>`:''}</div>
  ${!t.assignee&&!['submitted','approved'].includes(t.status)?'<div class="claim-banner"><span>Belum ada yang mengambil lagu ini.</span><button class="button secondary" id="claim">Ambil tugas ↗</button></div>':''}
  ${t.status==='submitted'?'<div class="review-status">✓ Hasil sudah masuk ke antrean Review masuk pengelola. Tidak perlu mengirim file terpisah.</div>':''}${t.status==='approved'?'<div class="review-status">✓ Anotasi disetujui dan siap diekspor.</div>':''}
  <div class="section-heading"><div><h3>Periksa label section</h3><small id="section-count"></small></div><button class="text-button" id="confirm-all" ${!canEdit?'disabled':''}>Konfirmasi semua prediksi ✓</button></div>
  <div class="section-scroll"><table class="section-table"><thead><tr><th>#</th><th>RENTANG WAKTU</th><th>PREDIKSI</th><th>LABEL FINAL</th><th>CATATAN</th><th>✓</th></tr></thead><tbody>${t.sections.map((s,i)=>`<tr data-section="${i}"><td><span class="section-number">${String(i+1).padStart(2,'0')}</span></td><td><button class="section-play" data-play-section="${i}" title="Putar section" ${!t.audio_ready?'disabled':''}>▶</button><span class="section-times">${fmtFine(s.start)} – ${fmtFine(s.end)}</span></td><td><span class="label-chip ${esc(s.label_initial)}">${esc(names[s.label_initial])}</span>${s.needs_review?'<span class="ambiguous" title="Prediksi perlu perhatian lebih">◌</span>':''}</td><td><select class="label-select" data-label="${i}" ${!canEdit?'disabled':''} aria-label="Label final section ${i+1}"><option value="">Belum diperiksa</option>${state.labels.map(l=>`<option value="${l}" ${s.label_manual===l?'selected':''}>${names[l]}</option>`).join('')}</select></td><td><input class="note-input" data-note="${i}" placeholder="Opsional…" maxlength="2000" value="${esc(s.note)}" ${!canEdit?'disabled':''} aria-label="Catatan section ${i+1}"></td><td><button class="row-confirm ${s.label_manual!==null?'checked':''}" data-confirm="${i}" title="Konfirmasi prediksi section ini" ${!canEdit?'disabled':''}>✓</button></td></tr>`).join('')}</tbody></table></div>
  <div class="editor-footer"><span id="save-state" class="save-indicator">✓ Tersimpan di cloud</span><div class="editor-actions">${canEdit?'<button class="button secondary" id="save">Simpan draf</button><button class="button primary" id="submit">Kirim untuk review ↗</button>':''}${admin()&&t.status==='submitted'?'<button class="button secondary" id="reopen">Kembalikan</button><button class="button primary" id="approve">Setujui anotasi ✓</button>':''}${admin()&&t.status==='approved'?'<button class="button secondary" id="reopen">Buka kembali</button>':''}${!canEdit&&!['submitted','approved'].includes(t.status)&&t.assignee!==state.me.id?'<span class="hint">Sedang dikerjakan anotator lain.</span>':''}</div></div>`;
  // CSP avoids inline script; timeline widths are assigned as properties below.
  $$('.timeline-segment').forEach((el,i)=>{el.removeAttribute('style');el.style.flex=String(t.sections[i].end-t.sections[i].start)});
  audio=$('#audio');
  if(audio&&state.authMode==='code'){
    const active=audio;$('#current-time').textContent='Memuat…';
    fetch('/api/tracks/'+t.id+'/audio',{headers:authHeaders()}).then(async r=>{if(!r.ok)throw new Error('Audio gagal dimuat. Periksa kode akses.');return r.blob()}).then(blob=>{if(audio!==active)return;blobURL=URL.createObjectURL(blob);active.src=blobURL;$('#current-time').textContent='0:00'}).catch(e=>notice(e.message,true));
  }
  if(audio){
    audio.addEventListener('timeupdate',onTime);
    audio.addEventListener('play',()=>$('#play-main').textContent='Ⅱ');
    audio.addEventListener('pause',()=>$('#play-main').textContent='▶');
    audio.addEventListener('ended',()=>{stopAt=null});
    audio.addEventListener('error',()=>notice('Audio tidak dapat diputar. Periksa apakah MP3 berhasil diunggah.',true));
    $('#play-main').onclick=()=>{stopAt=null;audio.paused?audio.play().catch(e=>notice(e.message,true)):audio.pause()};
    $('#rate').onchange=e=>audio.playbackRate=Number(e.target.value);
    $('#seek').oninput=e=>{stopAt=null;audio.currentTime=Number(e.target.value)};
  }
  $$('[data-play-section]').forEach(b=>b.onclick=()=>playSection(Number(b.dataset.playSection)));
  $$('[data-label]').forEach(el=>el.onchange=()=>{const i=Number(el.dataset.label);state.track.sections[i].label_manual=el.value||null;setDirty();$(`[data-confirm="${i}"]`).classList.toggle('checked',!!el.value)});
  $$('[data-note]').forEach(el=>el.oninput=()=>{state.track.sections[Number(el.dataset.note)].note=el.value;setDirty()});
  $$('[data-confirm]').forEach(b=>b.onclick=()=>{const i=Number(b.dataset.confirm),s=state.track.sections[i];s.label_manual=s.label_manual??s.label_initial;$(`[data-label="${i}"]`).value=s.label_manual;b.classList.add('checked');setDirty()});
  $('#confirm-all').onclick=()=>{if(!confirm('Konfirmasi bahwa Anda sudah mendengar dan memeriksa seluruh section. Section yang sudah dikoreksi tetap dipertahankan.'))return;for(let i=0;i<t.sections.length;i++){const s=t.sections[i];s.label_manual=s.label_manual??s.label_initial;$(`[data-label="${i}"]`).value=s.label_manual;$(`[data-confirm="${i}"]`).classList.add('checked')}setDirty()};
  if($('#claim'))$('#claim').onclick=()=>action('claim');
  if($('#assign-user'))$('#assign-user').onchange=e=>{if(state.dirty){notice('Simpan draf sebelum mengubah penugasan.',true);e.target.value=t.assignee??'';return}action('assign',{assignee:e.target.value})};
  if($('#save'))$('#save').onclick=()=>save(false);
  if($('#submit'))$('#submit').onclick=()=>save(true);
  if($('#approve'))$('#approve').onclick=()=>action('approve');
  if($('#reopen'))$('#reopen').onclick=()=>action('reopen');
  updateChecked();
}
function onTime(){
  if(!audio||!state.track)return;
  if(stopAt!==null&&audio.currentTime>=stopAt){audio.pause();stopAt=null}
  $('#current-time').textContent=fmt(audio.currentTime);$('#seek').value=audio.currentTime;
  $('#playhead').style.left=`${Math.min(100,audio.currentTime/state.track.duration*100)}%`;
  const index=state.track.sections.findIndex(s=>audio.currentTime>=s.start&&audio.currentTime<s.end);
  if(index!==state.section){$$('[data-section]').forEach(el=>el.classList.toggle('current',Number(el.dataset.section)===index));state.section=index}
}
function playSection(i){if(!audio)return;const s=state.track.sections[i];audio.currentTime=s.start;stopAt=s.end;audio.play().catch(e=>notice(e.message,true));}
async function action(verb,extra={}){
  if(state.busy)return;state.busy=true;
  try{const result=await api(`/tracks/${state.track.id}/${verb}`,{version:state.track.version,...extra});resetAudio();state.track=result;state.dirty=false;renderEditor();await refresh();notice(verb==='save'&&extra.submit?'Anotasi berhasil dikirim ke pengelola. Pengelola dapat membukanya di Review masuk.':'Perubahan tersimpan di cloud.')}catch(e){notice(e.message,true)}finally{state.busy=false}
}
async function save(submit){
  if(submit&&state.track.sections.some(s=>s.label_manual===null)){notice('Periksa seluruh section sebelum mengirim. Gunakan tombol ✓ untuk prediksi yang sudah benar.',true);return}
  await action('save',{edits:state.track.sections.map(s=>({label_manual:s.label_manual,note:s.note??''})),submit});
}
function renderTeam(){
  $('#share-links').innerHTML=state.shares.map(s=>`<div class="share-row"><strong>${esc(s.name)}</strong><span>${state.tracks.filter(t=>t.assignee===s.id).length} lagu</span><button class="text-button" data-copy-link="${esc(s.id)}" ${s.url?'':'disabled'}>${s.url?'Salin tautan ↗':'Akses dicabut'}</button></div>`).join('');
  $$('[data-copy-link]').forEach(b=>b.onclick=async()=>{const url=state.shares.find(s=>s.id===b.dataset.copyLink).url;try{await navigator.clipboard.writeText(url);notice('Tautan '+state.shares.find(s=>s.id===b.dataset.copyLink).name+' disalin.')}catch{prompt('Salin tautan undangan ini:',url)}});
  $('#team-count').textContent=`${state.users.length} anggota`;
  const current=new Set(state.users.map(u=>u.email));
  $('#team-list').innerHTML=state.users.map(u=>`<div class="team-member"><span class="avatar">${esc(u.name.slice(0,1).toUpperCase())}</span><div><strong>${esc(u.name)}</strong><small>${esc(!u.email||u.email.endsWith('@code.local')||u.email==='owner@local'?'Tautan pribadi':u.email)}</small></div><span class="badge">${u.role==='admin'?'Pengelola':'Anotator'}</span>${admin()&&state.authMode==='code'&&u.role!=='admin'?`<button class="text-button" data-revoke="${esc(u.id)}">Cabut kode</button>`:''}</div>`).join('')+state.invites.filter(i=>!current.has(i.email)).map(i=>`<div class="team-member"><span class="avatar">${esc(i.name.slice(0,1).toUpperCase())}</span><div><strong>${esc(i.name)}</strong><small>${esc(i.email)}</small></div><span class="badge">Belum masuk</span></div>`).join('');
}
$('#team-list').onclick=async e=>{const b=e.target.closest('[data-revoke]');if(!b||!confirm('Cabut akses anotator ini? Kode lama tidak dapat digunakan lagi.'))return;try{await api('/revoke',{id:b.dataset.revoke});notice('Kode akses dicabut. Hasil anotasi tetap tersimpan.')}catch(err){notice(err.message,true)}};
function view(which){$('#library-view').classList.toggle('hidden',which!=='library');$('#team-view').classList.toggle('hidden',which!=='team');$('#nav-review').classList.remove('active');$('#nav-library').classList.toggle('active',which==='library');$('#nav-team').classList.toggle('active',which==='team');$('#crumb-title').textContent=which==='library'?'Koleksi & anotasi':'Tim & hasil'}
$('#nav-review').onclick=()=>{view('library');state.filter='submitted';state.search='';$('#search').value='';$$('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter==='submitted'));$('#nav-review').classList.add('active');$('#nav-library').classList.remove('active');$('#crumb-title').textContent='Review masuk';renderList()};
$('#nav-library').onclick=()=>{view('library');state.filter='all';$$('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter==='all'));renderList()};$('#nav-team').onclick=()=>view('team');
$('#search').oninput=e=>{state.search=e.target.value.toLowerCase();renderList()};
$$('[data-filter]').forEach(b=>b.onclick=()=>{state.filter=b.dataset.filter;$$('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));renderList()});
$('#invite-form').onsubmit=async e=>{
  e.preventDefault();const form=e.target;const btn=form.querySelector('button');btn.disabled=true;
  try{const r=await api('/users',{name:form.elements.name.value,email:form.elements.email.value});notice(r.note);form.reset();await refresh()}catch(error){notice(error.message,true)}finally{btn.disabled=false}
};
$('#code-form').onsubmit=async e=>{e.preventDefault();const form=e.target;const btn=form.querySelector('button');btn.disabled=true;try{const r=await api('/codes',{name:form.elements.name.value});const el=$('#new-code');el.classList.remove('hidden');el.textContent='Simpan kode untuk '+r.name+' (hanya ditampilkan sekali):\n'+r.code;form.reset();await refresh()}catch(error){notice(error.message,true)}finally{btn.disabled=false}};
$('#code-login').onsubmit=async e=>{e.preventDefault();accessCode=e.target.elements.code.value.trim();try{state.me=await api('/me');sessionStorage.setItem('songstruct-code',accessCode);$('#auth-screen').classList.add('hidden');await boot()}catch(err){$('#code-error').textContent=err.message;accessCode=''}};
$('#download-upload-config').onclick=async()=>{try{const data=await api('/share-links');const blob=new Blob([JSON.stringify(data.upload_config,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='songstruct-upload.local.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);notice('Konfigurasi diunduh. Jalankan python3 upload_dataset.py di Mac.')}catch(e){notice(e.message,true)}};
$('#approve-all').onclick=async()=>{
  if(!admin()||state.busy)return;
  if(state.dirty){notice('Simpan draf yang sedang diedit terlebih dahulu.',true);return}
  const snapshot=state.tracks.filter(t=>t.status==='submitted'&&t.checked===t.total).map(t=>({id:t.id,version:t.version}));
  if(!snapshot.length)return;
  state.busy=true;const button=$('#approve-all');button.disabled=true;button.textContent='Menyetujui…';
  const selected=state.track?.id;
  try{
    const result=await api('/approve-all',{tracks:snapshot});
    await refresh();
    if(selected&&state.track?.id===selected&&!state.dirty&&snapshot.some(t=>t.id===selected)){const updated=await api('/tracks/'+selected);resetAudio();state.track=updated;renderEditor();}
    notice(result.approved+' lagu disetujui.'+(result.skipped?' '+result.skipped+' lagu dilewati karena sudah berubah atau belum lengkap.':'')+' Catatan dan label tetap tersimpan.');
  }catch(e){notice(e.message+' Muat ulang daftar untuk memeriksa status terbaru.',true);await refresh().catch(()=>{});}
  finally{state.busy=false;renderStats();}
};
$('#upload-open').onclick=()=>$('#upload-dialog').showModal();
$('#upload-run').onclick=async()=>{
  const jsonFiles=[...$('#json-files').files,...$('#json-folder').files].filter(f=>f.name.toLowerCase().endsWith('.json')),audioFiles=[...$('#audio-files').files,...$('#audio-folder').files].filter(f=>f.name.toLowerCase().endsWith('.mp3')),log=$('#upload-progress');
  if(!jsonFiles.length){log.textContent='Pilih setidaknya satu JSON hasil SongFormer.';return}
  const btn=$('#upload-run');btn.disabled=true;$('#upload-dialog .iconbutton').disabled=true;
  log.textContent='Memulai unggah…';let successes=0,failures=0;
  for(let i=0;i<jsonFiles.length;i++){
    const f=jsonFiles[i];
    try{
      log.textContent+=`\n[${i+1}/${jsonFiles.length}] ${f.name}`;log.scrollTop=log.scrollHeight;
      if(f.size>128*1024*1024)throw new Error('JSON lokal maksimal 128 MiB per lagu.');
      const document=JSON.parse(await f.text());
      const filename=String(document.track_id||document.audio_path||'').split(/[\\/]/).pop();
      const exact=audioFiles.filter(a=>document.track_id&&a.webkitRelativePath&&a.webkitRelativePath.endsWith('/'+document.track_id));
      const candidates=exact.length?exact:audioFiles.filter(a=>a.name===filename);
      if(candidates.length!==1)throw new Error(candidates.length?'Ada beberapa MP3 dengan nama sama. Unggah pasangan ini secara terpisah.':`MP3 pasangan tidak ditemukan: ${filename||'track_id kosong'}`);
      const audioFile=candidates[0];
      if(audioFile.size>90*1024*1024)throw new Error('MP3 melebihi batas 90 MB.');
      const payload={document:compactDocument(document),title:audioFile.name.replace(/\.mp3$/i,'')};
      if(new TextEncoder().encode(JSON.stringify(payload)).length>2_000_000)throw new Error('Payload sections setelah diringkas masih melebihi 2 MB.');
      const result=await api('/import',payload);
      const existing=await api('/tracks/'+result.id);
      if(existing.audio_ready){log.textContent+=' — sudah tersedia, dilewati.';continue}
      const res=await fetch('/api/tracks/'+result.id+'/audio',{method:'PUT',headers:{...authHeaders(),'content-type':'audio/mpeg','x-requested-with':'SongStruct'},body:audioFile});
      const reply=await res.json();if(!res.ok)throw new Error(reply.error||'Unggah audio gagal.');
      log.textContent+=' — selesai ✓';successes++;
    }catch(e){failures++;log.textContent+=' — '+e.message}
    log.scrollTop=log.scrollHeight;
  }
  log.textContent+=`\nSelesai: ${successes} audio baru, ${failures} gagal.`;btn.disabled=false;$('#upload-dialog .iconbutton').disabled=false;
  await refresh().catch(e=>notice(e.message,true));
};
window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue=''}});
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!['INPUT','TEXTAREA','SELECT','BUTTON'].includes(document.activeElement.tagName)&&audio&&!$('#upload-dialog').open){e.preventDefault();$('#play-main').click()}});
async function boot(){
  try{const config=await api('/config');state.authMode=config.auth_mode;state.managedTeam=config.managed_team;if(state.authMode==='code'){$('#chatgpt-login').classList.toggle('hidden',!config.hosted_sites);if(config.hosted_sites)$('#chatgpt-login').textContent='Masuk sebagai pengelola ↗';$('#auth-screen small').textContent='Satu kode untuk satu anotator. Hubungi pengelola jika belum mendapat kode.';$('#code-login').classList.remove('hidden');}state.me=await api('/me');$('#username').textContent=state.me.name;$('#role').textContent=admin()?'Pengelola koleksi':'Anotator';$('#avatar').textContent=state.me.name.slice(0,1).toUpperCase();if(admin())$$('.admin-only').forEach(el=>el.classList.remove('hidden'));else{$('#nav-team').classList.add('hidden');$('#nav-library-label').textContent='Tugas saya';$('#collection-title').textContent='Lagu bagian Anda';$('#page-title').innerHTML='Halo, '+esc(state.me.name)+'.<br><span>Ini bagian lagu Anda.</span>';$('#page-description').textContent='Hanya lagu yang ditugaskan kepada Anda yang ditampilkan. Setelah diperiksa, kirim langsung ke pengelola.';$('#stat-total').previousElementSibling.textContent='Total tugas saya';$$('[data-filter=mine],[data-filter=pending]').forEach(b=>b.classList.add('hidden'));}if(admin()&&state.authMode==='code'){$('#invite-form').classList.add('hidden');if(state.managedTeam){$('#share-panel').classList.remove('hidden');const sharing=await api('/share-links');state.shares=sharing.links;}else $('#code-form').classList.remove('hidden');}if(state.authMode==='code')$$('a[href^="/api/export"]').forEach(a=>a.onclick=async e=>{e.preventDefault();try{const r=await fetch(a.getAttribute('href'),{headers:authHeaders()});if(!r.ok)throw new Error('Ekspor gagal.');const blob=await r.blob(),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=a.href.includes('scope=all')?'songstruct-all.json':'songstruct-approved.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),10000)}catch(error){notice(error.message,true)}});await refresh();}
  catch(e){if([401,403].includes(e.status)){$('#auth-screen').classList.remove('hidden');$('#auth-message').textContent=e.message}else notice(e.message,true)}
}
boot();setInterval(()=>{if(state.me&&!document.hidden&&!state.busy)refresh().catch(()=>{})},20000);
