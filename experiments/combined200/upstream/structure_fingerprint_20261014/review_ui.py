import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import json,hashlib,base64
from pathlib import Path
import cv2,numpy as np
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8790';errors=[];checks={}
before=hashlib.sha256((ROOT/'catalog.json').read_bytes()).hexdigest()
def wait_match(page,button='matchStructure'):
    with page.expect_response(lambda r:r.url.endswith('/api/structure/match') and r.request.method=='POST',timeout=120000) as pending:
        page.locator('#'+button).click()
    response=pending.value;assert response.ok,response.text()
    page.wait_for_function('document.getElementById("matchStructure").disabled===false')
    return response.json()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--disable-gpu'])
    page=browser.new_page(viewport={'width':1480,'height':1080},device_scale_factor=1)
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE+'/structure_fingerprint.html',wait_until='networkidle')
    page.wait_for_function('window.structureView && structureView.ready && window.lastStructureResponse && !document.getElementById("matchStructure").disabled')
    assert page.locator('#querySample option').count()==8 and page.locator('#gallerySample option').count()==77
    original=page.evaluate('lastStructureResponse.fingerprint');page.locator('#appearance').select_option('hue');changed=wait_match(page)
    assert changed['fingerprint']==original and not changed['identity_confirmed'];checks['fixed_nodes_color_invariant']=True
    page.locator('#hideColor').check();page.locator('#showRelations').check();page.locator('#structureCanvas').screenshot(path=str(ROOT/'qa/geometry_only.png'))
    page.locator('#hideColor').uncheck();page.locator('#showRelations').uncheck();page.locator('#appearance').select_option('original')
    # Drag actual query node; movement must follow the pointer in both axes.
    box=page.locator('#structureCanvas').bounding_box();state=page.evaluate('({p:structureView.panels[0],n:structureView.query.nodes[0]})')
    x=box['x']+state['p']['x']+state['n']['x']*state['p']['scale'];y=box['y']+state['p']['y']+state['n']['y']*state['p']['scale']
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+14,y+10,steps=5);page.mouse.up()
    after=page.evaluate('structureView.query.nodes[0]');assert after['x']>state['n']['x'] and after['y']>state['n']['y']
    assert page.locator('#matchDownload').is_hidden();checks['drag_direction_correct']=True
    count=page.evaluate('structureView.query.nodes.length');page.locator('#editMode').select_option('add')
    page.mouse.click(box['x']+state['p']['x']+30,box['y']+state['p']['y']+50)
    assert page.evaluate('structureView.query.nodes.length')==count+1
    page.locator('#nodeRadius').fill('20');page.locator('#nodeRadius').dispatch_event('change')
    assert page.evaluate('structureView.query.nodes.at(-1).radius')==20
    page.locator('#removeNode').click();assert page.evaluate('structureView.query.nodes.length')==count
    page.locator('#resetNodes').click();page.locator('#editMode').select_option('move');checks['create_resize_delete_reset']=True
    response=wait_match(page);assert response['geometry_only'] and len(response['fingerprint']['normalized_nodes'])==count
    href=page.locator('#matchDownload').get_attribute('href');assert page.request.get(BASE+'/'+href).ok
    with page.expect_download() as download:page.locator('#downloadNodes').click()
    download.value.save_as(str(ROOT/'qa/browser_draft.json'));assert json.loads((ROOT/'qa/browser_draft.json').read_text(encoding='utf-8'))['query']['nodes']
    checks['downloads']=True
    # A manually moved gallery berry must travel with the exported evidence.
    page.locator('#editSide').select_option('gallery')
    page.locator('#structureCanvas').scroll_into_view_if_needed();box=page.locator('#structureCanvas').bounding_box()
    state=page.evaluate('({p:structureView.panels[1],n:structureView.gallery.nodes[0],key:structureView.gallery.key})')
    x=box['x']+state['p']['x']+state['n']['x']*state['p']['scale'];y=box['y']+state['p']['y']+state['n']['y']*state['p']['scale']
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+8,y+6,steps=4);page.mouse.up()
    result=wait_match(page,'comparePair');assert state['key'] in result['gallery_overrides']
    assert result['gallery_overrides'][state['key']][0]['x']>state['n']['x']
    page.locator('#resetNodes').click();page.wait_for_function('(x)=>Math.abs(structureView.gallery.nodes[0].x-x)<1e-6',arg=state['n']['x'])
    page.wait_for_function('document.getElementById("matchDownload").hidden===true');page.locator('#editSide').select_option('query')
    checks['gallery_edit_evidence_preserved']=True
    # Upload a real source image and circle its existing target manually by clicks.
    record=next(r for r in json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'))['records'] if r['key']=='GROW_02_00')
    sources=json.loads((ROOT.parent/'fingerprint_2d/all_records.json').read_text(encoding='utf-8'));source=sources[record['source_index']]
    image=cv2.imdecode(np.fromfile(source['image'],np.uint8),1);mask=cv2.imdecode(np.fromfile(source['mask'],np.uint8),0)
    outline=max(cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea)
    outline=cv2.approxPolyDP(outline,.005*cv2.arcLength(outline,True),True).reshape(-1,2)
    page.locator('.upload summary').click();page.locator('#structureFile').set_input_files(source['image'])
    page.wait_for_function('document.getElementById("uploadMessage").textContent.includes("0个顶点")')
    rect=page.locator('#uploadCanvas').bounding_box();resize=min(1,1600/max(image.shape[:2]));iw=round(image.shape[1]*resize);ih=round(image.shape[0]*resize)
    ratio=min(rect['width']/iw,rect['height']/ih);xx=(rect['width']-iw*ratio)/2;yy=(rect['height']-ih*ratio)/2
    for x,y in outline:page.mouse.click(rect['x']+xx+x*resize*ratio,rect['y']+yy+y*resize*ratio)
    with page.expect_response(lambda r:r.url.endswith('/api/structure/detect'),timeout=120000) as pending:page.locator('#detectStructure').click()
    result=pending.value;assert result.ok,result.text();assert len(result.json()['nodes'])>=6
    page.wait_for_function('structureView.query.key==="upload"');response=wait_match(page)
    assert len(response['candidates'])==4;checks['upload_polygon_detection_match']=True
    resized=cv2.resize(image,(iw,ih));image64=base64.b64encode(cv2.imencode('.jpg',resized)[1]).decode()
    legacy=page.request.post(BASE+'/api/fingerprint',data={'image_base64':image64,'polygon':(outline*resize).tolist(),'model_key':'legacy'},timeout=120000)
    assert legacy.ok,legacy.text();assert legacy.json()['dimension']==256
    checks['legacy_inference_after_structure_detection']=True
    # Inspect a known clear sample for final screenshot.
    page.locator('#querySample').select_option('FP3_03_00');page.wait_for_function('structureView.query.key==="FP3_03_00"')
    wait_match(page);page.locator('#structureCanvas').scroll_into_view_if_needed();page.screenshot(path=str(ROOT/'qa/structure_page.png'))
    page.locator('#structureCanvas').screenshot(path=str(ROOT/'qa/matching_canvas.png'))
    insufficient=page.request.post(BASE+'/api/structure/match',data={'nodes':[{'x':1,'y':2,'radius':3}]*3})
    assert insufficient.ok and insufficient.json()['status']=='insufficient';checks['insufficient_support']=True
    for path in ['/fingerprint.html','/fingerprint_mix.html','/']:
        assert page.request.get(BASE+path).ok
    assert not errors,errors;browser.close()
assert hashlib.sha256((ROOT/'catalog.json').read_bytes()).hexdigest()==before
checks.update(errors=errors,source_catalog_unchanged=True,chrome_gpu_disabled=True)
(ROOT/'qa/browser_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print('STRUCTURE_UI_PASS',json.dumps(checks),flush=True)
