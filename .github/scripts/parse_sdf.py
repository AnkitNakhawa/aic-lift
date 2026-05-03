import sys, re

data = sys.stdin.read()
# gz service returns proto text format: data: "<?xml ...>"
m = re.search(r'data:\s*"(.*)', data, re.DOTALL)
if m:
    sdf = m.group(1).encode().decode('unicode_escape').rstrip('"')
    open('/tmp/aic.sdf', 'w').write(sdf)
    print('SDF written, bytes:', len(sdf))
else:
    open('/tmp/aic.sdf', 'w').write(data)
    print('Fallback write, bytes:', len(data))
