"""Freeze before seeing held-out retrieval results. Refuse to overwrite a freeze."""
from pathlib import Path
import hashlib,json,datetime
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'synthetic_consumer'
target=OUT/'frozen_evaluation_protocol.json'
if target.exists():raise SystemExit('Protocol already frozen; do not overwrite after evaluation')
if (OUT/'evaluation_rankings.json').exists():raise SystemExit('Held-out rankings already opened; cannot claim a new freeze')
files=['consumer_engine.py','consumer_geometry.py','circle_network.py','evaluate_synthetic_consumer.py',
       'synthetic_consumer/decision_threshold.json','synthetic_consumer/detector_best.pt',
       'synthetic_consumer/query_manifest.json','synthetic_consumer/gallery.json']
data=dict(frozen_at=datetime.datetime.now().isoformat(),evaluation_rankings_not_opened=True,
          sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
target.write_text(json.dumps(data,indent=2),encoding='utf-8');print('PROTOCOL FROZEN',data['frozen_at'])
