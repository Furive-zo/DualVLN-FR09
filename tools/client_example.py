#!/usr/bin/env python3
import argparse
import json
import requests

ap=argparse.ArgumentParser()
ap.add_argument('--url',default='http://127.0.0.1:5801/infer')
ap.add_argument('--image',required=True)
ap.add_argument('--lookdown')
ap.add_argument('--episode-id',required=True)
ap.add_argument('--instruction',required=True)
ap.add_argument('--command',required=True,choices=['follow','turn_left','turn_right','safe_stop','stop'])
ap.add_argument('--reset',action='store_true')
a=ap.parse_args()
meta={'episode_id':a.episode_id,'instruction':a.instruction,'command':a.command,'reset':a.reset}
with open(a.image,'rb') as front:
    files={'front':front}
    look=open(a.lookdown,'rb') if a.lookdown else None
    if look: files['lookdown']=look
    try:
        response=requests.post(a.url,data={'json':json.dumps(meta,ensure_ascii=False)},files=files,timeout=180)
        print(response.status_code)
        print(json.dumps(response.json(),ensure_ascii=False,indent=2))
    finally:
        if look: look.close()
