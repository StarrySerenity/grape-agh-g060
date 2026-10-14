from pathlib import Path
root=Path(__file__).resolve().parent.parent
s=(root/'fingerprint.js').read_text(encoding='utf-8')
s=s.replace('let image=null,polygon=[],imageCanvas=', 'let modelKey="mixed",image=null,polygon=[],imageCanvas=')
s=s.replace('polygon});', 'polygon,model_key:modelKey});')
s=s.replace('button.disabled=true;', "button.disabled=true;$('modelSelect').disabled=true;")
s=s.replace('finally{button.disabled=false}', "finally{button.disabled=false;$('modelSelect').disabled=false}")
start=s.index('(async()=>{summary=')
s=s[:start]+'''async function setModel(key){
 $('queryFingerprint').disabled=true;try{modelKey=key;const registry=await json('fingerprint_mix_20261014/model_registry.json');
 const folder=key==='legacy'?'fingerprint_2d':registry.models[key].gallery;
 summary=await json(folder+'/summary.json');
 $('fpIdentity').replaceChildren(...summary.identities.filter(r=>!summary.identity_conflicts_excluded.includes(r.identity)).map(r=>new Option(`${r.identity} · ${r.images}视角`,r.identity)));
 $('fpIdentity').value=summary.identities.some(r=>r.identity==='GROW_02')?'GROW_02':$('fpIdentity').options[0].value;gallery();
 $('vectorsDownload').href=folder+'/all_embeddings.npy';
 $('identitiesDownload').href=folder+(key==='legacy'?'/summary.json':'/identity_fingerprints.json');
 $('syntheticVectorsDownload').hidden=key==='legacy';$('syntheticVectorsDownload').href=folder+'/all_synthetic_embeddings.npy';
 $('modelStatus').textContent=key==='legacy'?'原有模型 · 复用训练图库诊断':key==='mixed'?'实拍＋100串仿真 · 视频图库检索':'仅实拍对照 · 视频图库检索';
 $('matches').textContent='模型已切换，请重新提取指纹。';$('fpDownload').hidden=true;$('maskedInput').hidden=true;
 }finally{$('queryFingerprint').disabled=false}
}
$('modelSelect').onchange=()=>setModel($('modelSelect').value).catch(e=>{$('fpMessage').textContent=e.message});
(async()=>{
 const report=await json('fingerprint_mix_20261014/results.json');
 $('fpStats').innerHTML=[['真实可用身份',report.valid_real_identities],['独立仿真身份',report.synthetic_identities],['仿真图像',report.synthetic_images],['指纹维数',report.dimension]].map(([l,v])=>`<div><strong>${v}</strong>${l}</div>`).join('');
 const pct=v=>(v*100).toFixed(1)+'%';
 $('fpDiagnostics').innerHTML='<table><tr><th>模型</th><th>验证集身份首次命中率</th><th>测试集身份首次命中率</th><th>测试前三命中率</th><th>测试帧mAP</th></tr>'+[['real_only','仅实拍'],['mixed','实拍＋仿真']].map(([k,n])=>`<tr><td>${n}</td><td>${pct(report.validation[k].identity.top1)}</td><td>${pct(report.test[k].identity.top1)}</td><td>${pct(report.test[k].identity.top3)}</td><td>${pct(report.test[k].frame.map)}</td></tr>`).join('')+'</table>';
 $('testScope').textContent=`验证：6串、22张独立照片。测试：${report.test_identities.length}串、${report.test_queries}张独立照片，对46串视频图库检索。仿真数据是否有效以表中实际结果为准。`;
 if(report.test.mixed.identity.top1<=report.test.real_only.identity.top1)$('testScope').textContent+=' 本轮混合版未提高独立测试的首次命中率，作为实验对照保留；原有模型未替换。';
 $('modelSelect').value=report.selection.recommended_model;await setModel($('modelSelect').value);draw();
 window.fingerprintView={ready:true,identities:report.valid_real_identities,images:report.total_real_images,synthetic:report.synthetic_images};
})().catch(e=>{$('fpMessage').textContent=e.message;console.error(e)});
'''
(root/'fingerprint_mix.js').write_text(s,encoding='utf-8')
# Only add links; the existing page and its default model remain available.
for name in ['index.html','fingerprint.html']:
    p=root/name;t=p.read_text(encoding='utf-8')
    if 'fingerprint_mix.html' not in t:t=t.replace('<main>','<main><nav class="toolbar"><a href="fingerprint_mix.html">实拍＋仿真指纹实验 →</a></nav>',1)
    p.write_text(t,encoding='utf-8')
