"""Read-only source preservation check, for the original data machine."""
import argparse
import json
from pathlib import Path
from snapshot import digest

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manual-baseline',type=Path,required=True);args=parser.parse_args()
 root=Path(__file__).resolve().parent
 frozen=json.loads((root/'data/samples.json').read_text(encoding='utf-8'))
 baseline=json.loads(args.manual_baseline.read_text(encoding='utf-8-sig'))
 groups={}
 for label,mapping in [('frozen_sources',frozen['source_hashes']),('existing_manual_states',baseline)]:
  changed=[];missing=[]
  for path,expected in mapping.items():
   if not Path(path).exists():missing.append(path)
   elif digest(path)!=expected:changed.append(path)
  groups[label]={'checked':len(mapping),'changed':changed,'missing':missing,'passed':not changed and not missing}
 result={'groups':groups,'all_sources_unchanged':all(g['passed'] for g in groups.values())}
 (root/'evidence/source-preservation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=True))
 if not result['all_sources_unchanged']:raise SystemExit(1)
if __name__=='__main__':main()
