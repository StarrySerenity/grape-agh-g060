"""Exercise actual Chrome page, upload, geometry query, and loopback guards."""
import json,urllib.request,urllib.error
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'synthetic_consumer';BASE='http://127.0.0.1:8792'

def main():
 report=dict(browser='installed Chrome, headless, GPU disabled',errors=[],responses=[])
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--disable-gpu'])
  page=browser.new_page(viewport={'width':1440,'height':1120},device_scale_factor=1)
  page.on('pageerror',lambda e:report['errors'].append(str(e)))
  page.goto(BASE,wait_until='networkidle');assert page.locator('#samples option').count()==40
  for key,name in [('evaluation_090_00','known'),('evaluation_095_00','unknown')]:
   page.select_option('#samples',key)
   with page.expect_response(lambda r:r.url.endswith('/api/sample')) as response:
    page.click('#loadSample')
   detected=response.value.json();assert len(detected['nodes'])>=6
   page.wait_for_function('!document.querySelector("#match").disabled')
   with page.expect_response(lambda r:r.url.endswith('/api/match'),timeout=90000) as response:
    page.click('#match')
   matched=response.value.json()
   page.wait_for_function('!document.querySelector("#match").disabled',timeout=90000)
   assert len(matched['candidates'])==41 and matched['compared_views_per_identity']==72
   assert matched['uses_rgb_for_matching'] is False and matched['identity_confirmed'] is False
   report['responses'].append(dict(key=key,status=matched['status'],matched_identity=matched['matched_identity'],
                                  expected_identity=detected['expected_identity'],elapsed_seconds=matched['elapsed_seconds']))
   # Same JSON downloadable from the link and computed through the page.
   url=page.locator('#download').get_attribute('href')
   downloaded=json.loads(urllib.request.urlopen(BASE+url).read())
   assert downloaded['candidates'][0]['identity']==matched['candidates'][0]['identity']
   page.screenshot(path=str(OUT/'qa'/f'browser_{name}.png'),full_page=True)
   page.check('#geometry');page.screenshot(path=str(OUT/'qa'/f'browser_{name}_geometry.png'),full_page=True);page.uncheck('#geometry')
  report['random_samples']=[]
  for button,known in [('#newKnown',True),('#newUnknown',False)]:
   with page.expect_response(lambda r:r.url.endswith('/api/random'),timeout=30000) as response:page.click(button)
   random=response.value.json();assert len(random['nodes'])>=6
   assert random['expected_in_database']==known and random['matching_does_not_receive_identity']
   report['random_samples'].append(dict(seed=random['seed'],in_database=known,detected=len(random['nodes'])))
   page.wait_for_function('!document.querySelector("#match").disabled')
  page.locator('summary').click()
  page.set_input_files('#upload',str(OUT/'queries/evaluation_090_00.jpg'))
  page.wait_for_function('document.querySelector("#queryInfo").textContent.includes("点击")')
  box=page.locator('#queryCanvas').bounding_box()
  for x,y in [(.01,.01),(.99,.01),(.99,.99),(.01,.99)]:page.mouse.click(box['x']+box['width']*x,box['y']+box['height']*y)
  with page.expect_response(lambda r:r.url.endswith('/api/detect'),timeout=30000) as response:page.click('#detect')
  assert len(response.value.json()['nodes'])>=6
  page.wait_for_function('!document.querySelector("#match").disabled');report['upload_detection_passed']=True
  page.set_viewport_size({'width':390,'height':1000});page.wait_for_timeout(150)
  assert page.evaluate('document.documentElement.scrollWidth')<=390
  page.screenshot(path=str(OUT/'qa/browser_mobile.png'),full_page=True);report['mobile_overflow']=False
  browser.close()
 for name,headers in [('foreign_origin',{'Origin':'http://example.com'}),('foreign_host',{'Host':'example.com:8792'})]:
  try:urllib.request.urlopen(urllib.request.Request(BASE+'/health',headers=headers));raise AssertionError('Foreign access allowed')
  except urllib.error.HTTPError as e:assert e.code==403;report[name]=403
 malformed=urllib.request.Request(BASE+'/api/match',data=b'{"nodes":[{"x":0,"y":0,"radius":-1}]}',headers={'Content-Type':'application/json'})
 try:urllib.request.urlopen(malformed);raise AssertionError('Invalid circles accepted')
 except urllib.error.HTTPError as e:assert e.code==400;report['invalid_circle_status']=400
 assert not report['errors'];report['passed']=True
 (OUT/'browser_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
