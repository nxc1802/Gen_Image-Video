RECORDS=[{'file': 'minimax-h3/minimax-spoken-audio-native-dialogue-v2-3.json', 'sha256': '5fc12eaa083034840e69fb0146121714bcfdb5d6cbdf5c4d39d27ee129d48e91', 'expected': {'decoded_samples_sha256': '6dc18c82a29ae2ddabca217cd3a627fa1ed0dcbeda1c2c9bd05d31a8a2dbb5a0', 'duration_seconds': 4.48, 'input_sha256': '3f9ebf64622c92ab63901a0814c29d789fee638894252d3ab571f9d599c51b00', 'sample_rate': 16000, 'transcript': ' Factory makes useful tools.', 'verifier_pins_sha256': '17ba5ebdf6f9ba3e63ddf2bdc6fbf20925806ac9be2d317bd31740f4b7829067'}, 'mp4_sha256': '3f9ebf64622c92ab63901a0814c29d789fee638894252d3ab571f9d599c51b00'}, {'file': 'minimax-h3/minimax-spoken-audio-native-dialogue-v2-2.json', 'sha256': '5e7130b160f228072d33569beed53d31740cc4e326b95ff338a476c50615e39a', 'expected': {'decoded_samples_sha256': 'fdb42b8dec63710f632dde5dadeba4ef92056a87a66f9dae140485059e0e6601', 'duration_seconds': 4.48, 'input_sha256': '465555f56521f972bff9826f46301ca406551e4163a31131d48cf77c4e81e6b3', 'sample_rate': 16000, 'transcript': ' Factory makes useful tools.', 'verifier_pins_sha256': '17ba5ebdf6f9ba3e63ddf2bdc6fbf20925806ac9be2d317bd31740f4b7829067'}, 'mp4_sha256': '465555f56521f972bff9826f46301ca406551e4163a31131d48cf77c4e81e6b3'}, {'file': 'minimax-h3/minimax-spoken-audio-native-dialogue-v2-0.json', 'sha256': '0c4b6999583198fdcd12e45a7df49d585586d1c3ed153793aca3c645783b8a22', 'expected': {'decoded_samples_sha256': 'd86e6bfa0cecf0ea5b90bd0cdba78eaa160ceb17b6b8c2f31ce91feff7302bce', 'duration_seconds': 4.48, 'input_sha256': '259ee3fc8703dea1d74f088291b80b1b849fc3325d85e7ed0460ce4819b22674', 'sample_rate': 16000, 'transcript': ' Factory makes useful tools.', 'verifier_pins_sha256': '17ba5ebdf6f9ba3e63ddf2bdc6fbf20925806ac9be2d317bd31740f4b7829067'}, 'mp4_sha256': '259ee3fc8703dea1d74f088291b80b1b849fc3325d85e7ed0460ce4819b22674'}, {'file': 'minimax-h3/minimax-spoken-audio-native-dialogue-v2-1.json', 'sha256': '73c46d75b5ab8e44e1e47ee10d81affd2674f584c4db6f013db5b1a104a1bea0', 'expected': {'decoded_samples_sha256': '76d8227272d3a8a55fbe2c2c4b6436976758885948fa9fdbac4e4bd79846d06a', 'duration_seconds': 4.48, 'input_sha256': '078a39b4664b7a56b53b58a63b273c87ce492820eb7862f51a259cafc083a65e', 'sample_rate': 16000, 'transcript': ' Factory makes useful tools.', 'verifier_pins_sha256': '17ba5ebdf6f9ba3e63ddf2bdc6fbf20925806ac9be2d317bd31740f4b7829067'}, 'mp4_sha256': '078a39b4664b7a56b53b58a63b273c87ce492820eb7862f51a259cafc083a65e'}]

import base64,hashlib,io,json,platform,shutil,struct,subprocess,zipfile
from pathlib import Path
out=Path('/kaggle/working/audio_decoder_replay');out.mkdir()
report={'classification':'CPU_REDECODE_OF_EXISTING_REAL_MP4_NOT_NEW_MODEL_INFERENCE',
        'production_ready':False,'v3_certified':False,'full_factory_e2e_executed':False,
        'native_inference_executed':False,'platform':platform.platform(),'records':[],'status':'NOT_MET'}
try:
    archives=list(Path('/kaggle/input').rglob('native_wave_d.zip'))
    assert len(archives)==1,'Exactly one immutable receipt-bound source archive required'
    archive=zipfile.ZipFile(archives[0])
    decoder=Path(shutil.which('ffmpeg')).resolve()
    report['decoder']={'sha256':hashlib.sha256(decoder.read_bytes()).hexdigest(),
                       'version_output':subprocess.check_output([str(decoder),'-version'],text=True)}
    for index,record in enumerate(RECORDS):
        source=archive.read(record['file'])
        assert hashlib.sha256(source).hexdigest()==record['sha256'],'Wrong mounted native version'
        data=json.loads(source);raw=base64.b64decode(data['artifacts'][0]['data_base64'],validate=True)
        assert hashlib.sha256(raw).hexdigest()==record['mp4_sha256']
        argv=[str(decoder),'-nostdin','-v','error','-i','pipe:0','-map','0:a:0','-vn','-ac','1','-ar','16000','-f','f32le','pipe:1']
        decoded=subprocess.run(argv,input=raw,check=True,capture_output=True,timeout=60).stdout
        target=out/('actual_decoded_'+str(index)+'.f32le');target.write_bytes(decoded)
        sha=hashlib.sha256(decoded).hexdigest()
        row={'source_output_json_sha256':record['sha256'],'input_mp4_sha256':record['mp4_sha256'],
             'decoded_samples_sha256':sha,'sample_rate':16000,'sample_count':len(decoded)//4,
             'duration_seconds':len(decoded)/4/16000,'file':target.name,
             'matches_original_native_pcm':sha==record['expected']['decoded_samples_sha256'],
             'matches_original_native_duration':len(decoded)/4/16000==record['expected']['duration_seconds']}
        report['records'].append(row)
    report['status']='PASS' if all(r['matches_original_native_pcm'] and r['matches_original_native_duration'] for r in report['records']) else 'NOT_MET'
except Exception as exc:
    import traceback
    report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
finally:
    (out/'replay.json').write_text(json.dumps(report,indent=2)+'\n')
    with zipfile.ZipFile('/kaggle/working/audio_decoder_replay.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in out.iterdir():archive.write(path,path.name)
    print(json.dumps(report),flush=True)
if report['status']!='PASS':raise SystemExit(1)
