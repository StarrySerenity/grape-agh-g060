import { execFile } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { isAbsolute, join } from 'node:path'
import { promisify } from 'node:util'

const exec = promisify(execFile)
const operations = ['inventory', 'describe', 'retrieve', 'evaluate', 'forecast', 'report', 'latest', 'structure_evaluate', 'growth_evaluate', 'structure_case']

export async function invoke(operation, args, config) {
  if (!operations.includes(operation)) throw new Error('Unsupported operation')
  const allowed=operation==='structure_case'?['case_index','model_key']:['describe','retrieve'].includes(operation)?['sample_id']:[]
  if (Object.keys(args).some(key => !allowed.includes(key))) throw new Error('Unsupported argument')
  if(operation==='structure_case' && ((!Number.isInteger(args.case_index??0)) || (args.case_index??0)<0 || (args.case_index??0)>19 || !['real_only','mixed'].includes(args.model_key??'real_only')))throw new Error('Invalid structure case')
  if (['describe', 'retrieve'].includes(operation) && !/^[A-Za-z0-9_]{1,80}$/.test(args.sample_id ?? ''))
    throw new Error('Invalid sample ID')
  if (!isAbsolute(config.root) || !isAbsolute(config.python)) throw new Error('Runtime paths must be absolute')
  const argv = [join(config.root, 'cli.py'), operation]
  if (args.sample_id) argv.push('--sample-id', args.sample_id)
  if(operation==='structure_case')argv.push('--case-index',String(args.case_index??0),'--model-key',args.model_key??'real_only')
  const env = Object.fromEntries(Object.entries(process.env).filter(([name]) =>
    ['path', 'pathext', 'systemroot', 'windir', 'comspec', 'userprofile', 'localappdata', 'appdata', 'temp', 'tmp'].includes(name.toLowerCase())))
  Object.assign(env, { PYTHONIOENCODING: 'utf-8', PYTHONDONTWRITEBYTECODE: '1', OPENBLAS_NUM_THREADS: '2', OMP_NUM_THREADS: '2' })
  let stdout
  try {
    ({ stdout } = await exec(config.python, argv, { cwd: config.root, env, windowsHide: true, timeout: 120000, maxBuffer: 4*1024*1024 }))
  } catch (error) {
    if (error.stdout) {
      try { const result = JSON.parse(error.stdout); if (result.error) throw new Error(result.error) }
      catch (parsed) { if (!(parsed instanceof SyntaxError)) throw parsed }
    }
    throw new Error('Numerical tool process failed; inspect local runtime configuration')
  }
  const result = JSON.parse(stdout)
  if (result.ok !== true) throw new Error(result.error ?? 'Numerical tool refused input')
  return result
}

const Kind = Symbol.for('TypeBox.Kind')
export const grapeTools = {
  inject: ['extension'],
  apply(ctx) {
    const agnes = ctx.extension()
    for (const operation of operations) {
      const needsSample = ['describe', 'retrieve'].includes(operation)
      const parameters = { [Kind]: 'Object', type: 'object', additionalProperties: false,
        properties: needsSample ? { sample_id: { [Kind]: 'String', type: 'string', pattern: '^[A-Za-z0-9_]{1,80}$', maxLength:80 } } : {},
        required: needsSample ? ['sample_id'] : [] }
      if(operation==='structure_case')parameters.properties={case_index:{[Kind]:'Integer',type:'integer',minimum:0,maximum:19},model_key:{[Kind]:'String',type:'string',enum:['real_only','mixed']}}
      const descriptions = {
        inventory: 'List the frozen grape dataset, identity exclusions, candidate sample IDs and limitations. Read-only.',
        describe: 'Compute dimensionless structural descriptors and quality notes for a frozen sample ID. Read-only.',
        retrieve: 'Rank independent capture records using fixed structural fingerprint features; return candidates and reference rank. Read-only. Not a verified identity decision.',
        evaluate: 'Compute real identity retrieval metrics and date-specific scopes. Read-only, fixed settings.',
        forecast: 'Evaluate relative morphology forecasts on ten longitudinal identities using leave-one-identity-out Ridge and mean baseline. Read-only. Does not predict millimeters or actual growth rates.',
        report: 'Recompute numerical results, plots and report data in the project results directory. Writes only generated reports; does not modify source models.',
        latest:'Read the dated October 14 snapshot overview and current demonstration links. Read-only.',
        structure_evaluate:'Recompute development and old-photo regression retrieval scores from per-query records for real-only and synthetic-assisted 128D geometry networks. Not new independent test accuracy.',
        growth_evaluate:'Recompute GROW and YG endpoint morphology MAEs from held-out folds and verify test identity exclusion. Animation and default 1.35 expansion are illustration settings.',
        structure_case:'Actually encode one frozen regression query with the trained 128D geometry network and rank the frozen video gallery. case_index 0..19; model_key real_only or mixed. No pixels or color enter the encoder. Does not confirm identity.' }
      agnes.registerTool({ name: 'grape_' + operation, description: descriptions[operation], parameters,
        meta: { isReadOnly: operation !== 'report', isDestructive: false, isConcurrencySafe: operation !== 'report',
          isOpenWorld: false, replay: operation === 'report' ? 'idempotent' : 'safe',
          costHint:undefined, deferLoading:false, requiresApproval: operation === 'report' ? 'always' : 'never' },
        async execute(args) {
          try {
            const config = JSON.parse(await readFile(new URL('./runtime-config.json', import.meta.url), 'utf8'))
            const result = await invoke(operation, args, config)
            return { content: [{type:'text',text:JSON.stringify(result)}], structured: result }
          } catch (error) {
            const result = {ok:false,error:error.message}
            return { content: [{type:'text',text:JSON.stringify(result)}], structured:result, isError:true }
          }
        } })
    }
  }
}
