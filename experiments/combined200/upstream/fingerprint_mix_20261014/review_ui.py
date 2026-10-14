"""Exercise the actual upload/polygon/model-switch/download flow in CPU Chrome."""
import json,math
from pathlib import Path
import cv2,numpy as np
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'real_rows.json').read_text(encoding='utf-8'))
split=json.loads((ROOT/'real_protocol.json').read_text(encoding='utf-8'))
row=rows[split['test_query_indices'][0]];qa=ROOT/'qa';errors=[];responses=[]
im=cv2.imdecode(np.fromfile(row['image'],np.uint8),cv2.IMREAD_COLOR)
mask=cv2.imdecode(np.fromfile(row['mask'],np.uint8),cv2.IMREAD_GRAYSCALE)
contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
outline=max(contours,key=cv2.contourArea);outline=cv2.approxPolyDP(outline,.005*cv2.arcLength(outline,True),True).reshape(-1,2)
scale=min(1,1600/max(im.shape[:2]));polygon=(outline*scale).tolist()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--disable-gpu'])
    page=browser.new_page(viewport={'width':1450,'height':1050},device_scale_factor=1)
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8790/fingerprint_mix.html',wait_until='networkidle')
    page.wait_for_function('window.fingerprintView && fingerprintView.ready')
    assert page.locator('#fpIdentity option').count()==46
    page.locator('#imageFile').set_input_files(row['image'])
    page.wait_for_function("document.getElementById('polygonStatus').textContent.includes('0 个顶点')")
    canvas=page.locator('#fpCanvas').bounding_box()
    iw=round(im.shape[1]*scale);ih=round(im.shape[0]*scale)
    ratio=min(canvas['width']/iw,canvas['height']/ih);x=(canvas['width']-iw*ratio)/2;y=(canvas['height']-ih*ratio)/2
    for px,py in polygon:page.mouse.click(canvas['x']+x+px*ratio,canvas['y']+y+py*ratio)
    for key in ['mixed','real_only','legacy','mixed']:
        page.locator('#modelSelect').select_option(key)
        page.wait_for_function("document.getElementById('matches').textContent.includes('模型已切换')")
        with page.expect_response(lambda r:r.url.endswith('/api/fingerprint') and r.request.method=='POST',timeout=120000) as pending:
            page.locator('#queryFingerprint').click()
        response=pending.value;assert response.ok,response.text();data=response.json()
        assert data['model_key']==key and len(data['fingerprint'])==256 and len(data['candidates'])==5
        assert np.isfinite(data['fingerprint']).all() and abs(np.linalg.norm(data['fingerprint'])-1)<1e-5
        page.wait_for_function("document.getElementById('fpDownload').hidden===false")
        page.wait_for_function("Array.from(document.querySelectorAll('#matches img')).every(i=>i.complete&&i.naturalWidth>0)")
        download=page.request.get('http://127.0.0.1:8790/'+page.locator('#fpDownload').get_attribute('href'))
        assert download.ok and download.json()['model_key']==key
        responses.append(dict(model=key,query_identity=row['identity'],top=data['candidates'][0]['identity'],dimension=data['dimension']))
    page.screenshot(path=str(qa/'query_demo.png'),full_page=False)
    page.locator('#fpDiagnostics').screenshot(path=str(qa/'comparison_table.png'))
    assert page.request.get('http://127.0.0.1:8790/'+page.locator('#vectorsDownload').get_attribute('href')).ok
    page.goto('http://127.0.0.1:8790/fingerprint.html',wait_until='networkidle')
    page.wait_for_function('window.fingerprintView && fingerprintView.ready');assert page.locator('#fpIdentity option').count()==50
    page.goto('http://127.0.0.1:8790/',wait_until='networkidle');assert page.locator('a[href="fingerprint_mix.html"]').count()==1
    assert not errors,errors;browser.close()
(qa/'browser_checks.json').write_text(json.dumps(dict(errors=errors,queries=responses,upload=True,polygon_clicks=len(polygon),
    download=True,old_page_preserved=True,thumbnail_alignment=True),indent=2),encoding='utf-8')
print('FINGERPRINT_UI_PASS',json.dumps(responses),flush=True)
