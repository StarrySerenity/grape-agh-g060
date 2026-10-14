import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { pathToFileURL, fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
const root=dirname(fileURLToPath(import.meta.url))
const aghSource=process.env.AGH_SOURCE ?? 'D:/Tools/agnes-harness'
const aghHome=process.env.AGH_HOME ?? 'D:/Tools/agnes-grape/home'
const {createClient,memoryJournal}=await import(pathToFileURL(join(aghSource,'packages/sdk/src/index.node.ts')).href)
const owner=JSON.parse(await readFile(join(aghHome,'data/daemon/owner.json'),'utf8'))
const client=createClient({transport:{kind:'unix',path:owner.socketPath,serverIdentity:{pid:owner.pid,processStartId:owner.processStartId}},auth:{kind:'local'},journal:memoryJournal()})
await mkdir(join(root,'evidence'),{recursive:true})
const failure=process.argv.includes('--failure')
const correction=process.argv.includes('--correct')
const boundary=process.argv.includes('--boundary')
const events=[],calls=new Map()
const prompt=failure ? '请只调用 grape_describe，sample_id 为 unknown，验证不存在样本的失败路径。工具返回错误后解释原因，不使用别的工具、不编造结果、不修改代码。' : `完成 G060 葡萄项目专业分析任务。只能调用 grape_ 开头的专用工具，不使用 exec/edit/write，不重新拟合或重建，不覆盖手动状态。
必须按顺序执行，读取前一步返回后再决定下一步，不要一次并行发出全部调用：
1. grape_inventory 核对冻结样本、身份冲突和适用范围。
2. grape_describe 读取 May_g2_07 的几何指标，然后 grape_retrieve 检索它的独立拍摄候选。说明工具有没有命中真实身份。
3. grape_evaluate 验证整体、同日期和跨日期检索，不根据单个样本夸大能力。
4. grape_forecast 计算10个身份留一法预测，与均值基线比较。无量纲指标不等于真实物理生长。
5. grape_report 生成数值报告，仅写本作品 results 目录。
6. 根据工具实际返回，中文总结已实现工程能力、真实实验结果与当前失败之处。若预测不优于基线必须明确说明，不能称精准防伪或准确生长预测。不需要询问是否继续，上述报告写入已获用户授权。`
let session,outcome,model
const correctionPrompt='请更正上轮解读：baseline 是8维确定性几何摘要的欧氏距离检索，不是随机猜测；没有显著性检验，不能说显著优于随机。请重新调用 grape_evaluate 和 grape_forecast，根据返回写一段完整中文结论。必须清楚写47个有效查询基础摘要Top1=1/47，结构Top1=3/47、Top3=8/47；跨日期结构Top1=0/47，Top3=1/47；预测两个指标MAE都高于训练均值基线。无量纲结果不等于真实生长，身份相似度不等于防伪。保留工程可复现价值。只用这两个工具，不修改代码。'
try{
 await client.initialize()
 await client.workspace.add(root)
 const cfg=await client.config.get(),account=cfg.accounts.find(a=>a.accountId===cfg.defaultAccountId)
 if(!account||account.providerId!=='agnes-ai')throw new Error('Agnes default account is required')
 model=account.model
 session=correction ? await client.session.load(JSON.parse(await readFile(join(root,'evidence/professional-task.json'),'utf8')).sessionId,{cwd:root}) : await client.session.new({cwd:root,sessionKey:'agnes:local:local-dev:g060:'+ (failure?'failure':'professional')+':'+Date.now()})
 await session.setModel({slot:'primary',route:account.route,model})
 session.onPermissionRequest(async request=>{
  const call=request.toolCall,name=String(call._meta?.['ai.agnes.harness']?.tool??call.name??call.title??call.toolName??'')
  const allowed=name==='grape_report'
  console.log('Approval:',name,allowed?'allow generated report once':'deny outside task')
  const option=request.options.find(x=>x.kind===(allowed?'allow_once':'reject_once'))
  return option?{optionId:option.optionId}:{verdict:'rejected'}
 })
 session.listeners.add((method,params)=>{
  const event=params.event
  if(!event)return
  events.push(event)
  if(event.type==='tool/call'){
   calls.set(event.data.toolUseId,{name:event.data.name,args:event.data.args,callSeq:event.seq})
   console.log('Tool:',event.data.name)
  }
  if(event.type==='tool/result'){
   const call=calls.get(event.data.toolUseId)
   if(call){call.isError=!!event.data.isError;call.resultSeq=event.seq
    try{call.summary=JSON.parse(event.data.content?.find(x=>x.type==='text')?.text??'{}')}catch{call.summary={parse_error:true}}
   }
   console.log('Tool returned:',event.data.isError?'error':'result')
  }
  if(event.type==='assistant/message')console.log('Assistant:',JSON.stringify(event.data.content).slice(0,6500))
 })
 await session.attach({filter:{acpUpdates:true,preview:true}})
 const prefix=correction?'correction':boundary?'boundary':failure?'failure':'professional'
 const activePrompt=correction?correctionPrompt:boundary?'请顺序验证身份冲突的边界：先用 grape_describe 读取 July_g3_05，确认 identity_excluded 与几何估计指标；再用 grape_retrieve 尝试此样本，读取工具拒绝结果。解释为什么仍可查看几何但不能进行身份检索。只用这两个专业工具，不编辑文件、不改变标签、不编造结果。':prompt
 await writeFile(join(root,'evidence',prefix+'-task.json'),JSON.stringify({sessionId:session.id,model,prompt:activePrompt},null,2))
 console.log('AGH session:',session.id,'model:',model)
 outcome=await session.prompt(activePrompt,{titleLocale:'zh-CN',signal:AbortSignal.timeout(600000)})
 const summary={sessionId:session.id,model,completed:outcome.reason==='completed',outcome,
  professional_steps:[...calls.values()].filter(c=>c.name.startsWith('grape_')&&!c.isError&&c.resultSeq).length,calls:[...calls.values()]}
 await writeFile(join(root,'evidence',prefix+'-events.jsonl'),events.map(e=>JSON.stringify(e)).join('\n')+'\n')
 await writeFile(join(root,'evidence',prefix==='professional'?'agent-summary.json':prefix+'-summary.json'),JSON.stringify(summary,null,2))
 console.log('Final task status:',JSON.stringify(outcome),'professional steps:',summary.professional_steps)
 if(!failure&&!correction&&!boundary&&summary.professional_steps<3)throw new Error('Fewer than three verified professional tool steps')
}finally{client.close()}
