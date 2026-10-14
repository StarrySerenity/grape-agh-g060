import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { pathToFileURL, fileURLToPath } from 'node:url'
import { join, dirname } from 'node:path'
import { randomUUID } from 'node:crypto'
const aghSource=process.env.AGH_SOURCE ?? 'D:/Tools/agnes-harness'
const aghHome=process.env.AGH_HOME ?? 'D:/Tools/agnes-grape/home'
const {createClient,memoryJournal}=await import(pathToFileURL(join(aghSource,'packages/sdk/src/index.node.ts')).href)
const owner=JSON.parse(await readFile(join(aghHome,'data/daemon/owner.json'),'utf8'))
const client=createClient({transport:{kind:'unix',path:owner.socketPath,serverIdentity:{pid:owner.pid,processStartId:owner.processStartId}},auth:{kind:'local'},journal:memoryJournal()})
const out=join(dirname(fileURLToPath(import.meta.url)),'evidence')
await mkdir(out,{recursive:true})
const operations=[]
try{
 await client.initialize()
 const base={profile:'local-dev',clientId:await client.clientId()}
 const source={type:'file',ref:process.env.GRAPE_PLUGIN_SOURCE ?? 'file:./.agh-grape-tools-g060'}
 const wait=async receipt=>{
  const deadline=Date.now()+90000
  while(Date.now()<deadline){
   const op=await client.packages.operation.get({profile:base.profile,operationId:receipt.operationId})
   if(['completed','failed','cancelled','rolled-back'].includes(op.state)){
    operations.push(op);console.log('Package operation:',op.kind??op.operation,op.state)
    if(op.state!=='completed')throw new Error(JSON.stringify(op.error??op))
    return op
   }
   await new Promise(resolve=>setTimeout(resolve,250))
  }
  throw new Error('Package operation timed out')
 }
 const preview=await wait(await client.packages.inspect({...base,source,commandId:randomUUID()}))
 console.log('Preview:',JSON.stringify(preview.preview))
 if(preview.preview?.blockers?.length)throw new Error('Plugin blockers must be resolved')
 const integrity=preview.preview.integrity
 const installing={...base,source,expectedIntegrity:integrity,commandId:randomUUID()}
 await wait(process.argv.includes('--update') ? await client.packages.update({...installing,id:'@g060/grape-tools'}) : await client.packages.install(installing))
 await wait(await client.packages.trust({...base,id:'@g060/grape-tools',expectedIntegrity:integrity,capabilityHash:preview.preview.capabilityHash,commandId:randomUUID()}))
 await wait(await client.packages.enable({...base,id:'@g060/grape-tools',expectedInstalledIntegrity:integrity,commandId:randomUUID()}))
 console.log('Installed grape plugin:',JSON.stringify(await client.packages.list({profile:base.profile})))
}finally{await writeFile(out+(process.argv.includes('--update')?'/plugin-update.json':'/plugin-install.json'),JSON.stringify(operations,null,2));client.close()}
