import json,glob,os,sys
w=[]
for wk in sys.argv[1].split(','):
    if 'games' in sys.argv[2:] or len(sys.argv)==2:
        for f in sorted(glob.glob(f'out/2026-w{wk}-*.json')):
            w.append({"op":"set","collection":"games","doc_id":os.path.basename(f)[:-5],"file_path":os.path.abspath(f)})
    w.append({"op":"set","collection":"bulk","doc_id":f"2026-w{wk}","file_path":os.path.abspath(f'out/bulk-2026-w{wk}.json')})
print(json.dumps(w))
