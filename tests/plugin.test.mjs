import assert from 'node:assert/strict'
import test from 'node:test'
import { existsSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

test('tool runner only accepts fixed commands and bounded sample IDs', async () => {
  const { invoke } = await import('../plugin/index.mjs')
  const preferred='D:/Anaconda/envs/grape1/python.exe'
  const python=process.env.GRAPE_PYTHON ?? (existsSync(preferred)?preferred:execFileSync('python',['-c','import sys; print(sys.executable)'],{encoding:'utf8'}).trim())
  const config = { python, root: fileURLToPath(new URL('..',import.meta.url)) }
  await assert.rejects(() => invoke('exec', { command: 'anything' }, config), /Unsupported/)
  await assert.rejects(() => invoke('describe', { sample_id: '../../credentials' }, config), /sample/)
  const data = await invoke('inventory', {}, config)
  assert.equal(data.evaluation_samples, 73)
  assert.equal(data.evaluation_identities, 46)
  await assert.rejects(() => invoke('describe', { sample_id: 'unknown' }, config), /Unknown sample/)
})

test('registered tools provide the complete AGH metadata contract', async () => {
  const { grapeTools } = await import('../plugin/index.mjs')
  const definitions=[]
  grapeTools.apply({ extension: () => ({ registerTool: definition => definitions.push(definition) }) })
  assert.equal(definitions.length,10)
  for(const definition of definitions) {
    assert.ok(Object.hasOwn(definition.meta,'costHint'),'AGH requires an explicit costHint field even when undefined')
    assert.equal(definition.meta.isReadOnly,definition.name!=='grape_report')
  }
})
