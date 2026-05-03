import sys
import re

data = sys.stdin.read()
print(f"gz service output: {len(data)} bytes")
print(f"First 300 chars: {repr(data[:300])}")

m = re.search(r'data:\s*"(.*)', data, re.DOTALL)
if m:
    sdf = m.group(1).encode().decode("unicode_escape").rstrip('"').strip()
    open("/tmp/aic.sdf", "w").write(sdf)
    print(f"SDF written: {len(sdf)} bytes")
else:
    print("WARNING: no 'data:' field found in gz service output")
    open("/tmp/aic.sdf", "w").write(data)
    print(f"Fallback write: {len(data)} bytes")

content = open("/tmp/aic.sdf").read(100).strip()
if not content.startswith("<"):
    print(f"ERROR: /tmp/aic.sdf is not valid XML. First 200 chars: {repr(content[:200])}")
    sys.exit(1)

print("SDF validation OK")
