import sys

sys.path.append("/data3/haoyunx/work/utils/model_inference")
from gradio_infer.base.tgi_infer import process_openai
from flask_cors import *
from flask import request, Flask, Response
import logging
from text_generation import Client
import datetime as dt
import json
from datetime import datetime

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)

model_path = "/data2/haoyun/work/models/z4/tigerbot-70b-sft-discharge-train3-step1500-1epoch"
processor = process_openai.Processor(model_path)
tgi_api = "101.69.162.5:8301"
client = Client(f"http://{tgi_api}", timeout=120)


def tgi_prepare(gather_dict_list, diagnosis_at_discharge):
    instruction = ""
    instruction += "病程：" + "\n"

    for idx, line in enumerate(gather_dict_list):
        DOC_TIME = line['doc_time']
        diagnosis = line['diagnosis']
        check_test_results = line['check_test_results']
        physical_examination = line['physical_examination']
        chief_complaints_of_patients = line['chief_complaints_of_patients']
        disease_analysis_diagnosis_treatment_plan = line['disease_analysis_diagnosis_treatment_plan']
        instruction += "(病程记录" + str(idx + 1) + ")" + "\n"
        instruction += "病程记录时间：" + DOC_TIME + "\n"
        instruction += "病人主诉：" + chief_complaints_of_patients + "\n"
        instruction += "查体：" + physical_examination + "\n"
        instruction += "检查检验结果：" + check_test_results + "\n"
        instruction += "诊断：" + diagnosis + "\n"
        instruction += "病情分析与诊疗计划：" + disease_analysis_diagnosis_treatment_plan + "\n\n"

    instruction += "\n"
    instruction += "出院诊断：" + "\n"
    instruction += diagnosis_at_discharge

    # print(instruction)

    messages = []
    messages.append({"role": "user", "content": instruction})

    inputs = processor.preprocess(
        messages=messages,
        do_sample=True,
        top_p=None,
        temperature=0.3,
        max_input_length=8192,
        max_output_length=4096
    )

    return inputs

# 后端代码加在这
def backend_part(req_json):
    mrn = req_json['mrn']
    series = req_json['series']
    ...

@app.route("/infer", methods=["POST", "GET"])
def infer():
    req_json = request.json

    data_json = backend_part(req_json)

    patient_name = data_json['name']
    sex = data_json['sex']
    age = data_json['age']

    # 1. 入院诊断
    RYZD_interface_value_list = data_json['z4_interface_output']['RYZD']
    RYZD_text = ""
    for idx, CYZD_interface in enumerate(RYZD_interface_value_list):
        RYZD_interface_value = CYZD_interface['cbzd']['value']
        RYZD_text += str(idx + 1) + '.' + RYZD_interface_value + ' '
    RYZD_text = RYZD_text.strip()

    # 2. 出院诊断
    CYZD_interface_value_list = data_json['z4_interface_output']['CYZD']
    CYZD_text = ""
    for idx, CYZD_interface in enumerate(CYZD_interface_value_list):
        CYZD_interface_value = CYZD_interface['zd']['value']
        CYZD_text += str(idx + 1) + '.' + CYZD_interface_value + ' '
    CYZD_text = CYZD_text.strip()

    # 3. 住院天数
    # 入院日期
    admission_date = data_json['admission_date']

    # 出院日期
    discharge_data = data_json['discharge_data']

    # 计算住院天数 # 要改
    # length_of_stay = discharge_data - admission_date

    date1 = dt.datetime.strptime(discharge_data, "%Y-%m-%d").date()
    date2 = dt.datetime.strptime(admission_date, "%Y-%m-%d").date()

    length_of_stay = (date1 - date2).days

    # 4. 入院情况（首程["诊断依据"]）
    SCBCJL_ZDYJ = data_json['SCBCJL'][0]['zdyj']['value']
    # print(SCBCJL_ZDYJ)

    # 5. 住院经过
    # 检查
    hospitalization_check_list = data_json['z4_interface_output']['JC']
    hospitalization_check_list.sort(key=lambda k: datetime.strptime(k['rq']['value'].strip(), "%Y-%m-%d %H:%M:%S"),
                                    reverse=False)

    hospitalization_check_text = ""
    for hospitalization_check in hospitalization_check_list:
        hospitalization_check_item = hospitalization_check['jcxm']['value'].strip()
        hospitalization_check_result = hospitalization_check['jcsj']['value'].replace("\n", '').strip()
        if hospitalization_check_result[-1] != '。':
            hospitalization_check_result += '。'

        hospitalization_check_data = hospitalization_check['rq']['value'].strip()
        hospitalization_check_text += "(" + hospitalization_check_data + ")" + " " + hospitalization_check_item + ": " + hospitalization_check_result

    # print(hospitalization_check_text)

    hospitalization_test_list = data_json['z4_interface_output']['JY']
    hospitalization_test_text = ""
    hospitalization_test_dict = {}
    for hospitalization_test in hospitalization_test_list:
        hospitalization_test_item_name = hospitalization_test['xmmc']['value'].strip()
        hospitalization_test_item_result = hospitalization_test['xmjg']['value'].strip()
        hospitalization_test_item_dw = hospitalization_test['dw']['value'].strip()
        hospitalization_test_item_sfyc = hospitalization_test['sfyc']['value'].strip()
        hospitalization_test_time = hospitalization_test['sj']['value'].strip()
        hospitalization_test_broad_category_name = hospitalization_test['jcmc']['value'].strip()
        if "H" in hospitalization_test_item_sfyc:
            hospitalization_test_item_symbol = ' ↑'
        elif 'L' in hospitalization_test_item_sfyc:
            hospitalization_test_item_symbol = ' ↓'
        else:
            hospitalization_test_item_symbol = ''

        if hospitalization_test_item_symbol == '' and '中性粒细胞' not in hospitalization_test_item_name and '白细胞' not in hospitalization_test_item_name and '红细胞' not in hospitalization_test_item_name and '血小板' not in hospitalization_test_item_name:
            continue
        else:
            if hospitalization_test_broad_category_name not in hospitalization_test_dict.keys():
                hospitalization_test_dict[hospitalization_test_broad_category_name] = []

            hospitalization_test_dict[hospitalization_test_broad_category_name].append(
                {"hospitalization_test_item_name": hospitalization_test_item_name,
                 "hospitalization_test_item_result": hospitalization_test_item_result,
                 "hospitalization_test_item_dw": hospitalization_test_item_dw,
                 "hospitalization_test_item_sfyc": hospitalization_test_item_sfyc,
                 "hospitalization_test_time": hospitalization_test_time,
                 "hospitalization_test_item_symbol": hospitalization_test_item_symbol})

    hospitalization_test_dict_list = sorted(hospitalization_test_dict.items(), key=lambda k: datetime.strptime(
        k[1][0]['hospitalization_test_time'].strip(), "%Y-%m-%d %H:%M:%S"))
    for hospitalization_test_dict_line in hospitalization_test_dict_list:
        k = hospitalization_test_dict_line[0]
        v_list = hospitalization_test_dict_line[1]

        hospitalization_test_text += f"({v_list[0]['hospitalization_test_time']}){k}："
        for v in v_list:
            hospitalization_test_text += v['hospitalization_test_item_name'] + " " + v[
                'hospitalization_test_item_result'] + v['hospitalization_test_item_dw'] + v[
                                             'hospitalization_test_item_symbol'] + '，'

        hospitalization_test_text = hospitalization_test_text[:-1] + '; '

    # 检验
    # 病程经过、健康教育、随访计划
    course_key_list = ['ZZYSCFJL', 'ICULHCFJL', 'ZRYSCFJL', 'KZRCFJL', 'RCBCJL']
    gather_dict_list = []

    for course_key in course_key_list:
        course_list = data_json[course_key]
        for course in course_list:
            temp_dict = {}
            temp_dict['doc_time'] = course['doc_time']['value']
            temp_dict['chief_complaints_of_patients'] = course['brzs']['value']
            temp_dict['physical_examination'] = course['ct']['value']
            temp_dict['check_test_results'] = course['jcjyjg']['value']
            temp_dict['diagnosis'] = course['zd']['value']
            temp_dict['disease_analysis_diagnosis_treatment_plan'] = course['bqfxyzljh']['value']

            gather_dict_list.append(temp_dict)

    gather_dict_list.sort(key=lambda k: datetime.strptime(k['doc_time'], "%Y-%m-%d %H:%M:%S.%f"), reverse=False)

    # 6. 出院情况 (病程最后一天的主诉+查体)
    discharge_status = gather_dict_list[-1]['chief_complaints_of_patients'] + gather_dict_list[-1][
        'physical_examination']

    # 7. 出院医嘱
    order_of_discharge_list = data_json['z4_interface_output']['CYYZ']
    order_of_discharge_text = ""
    for order_of_discharge in order_of_discharge_list:
        yp = order_of_discharge['yp']['value']
        gg = order_of_discharge['gg']['value']
        yf = order_of_discharge['yf']['value']
        pd = order_of_discharge['pd']['value']
        order_of_discharge_text += yp + ' ' + gg + ' ' + yf + ' ' + pd + ', '

    inputs = tgi_prepare(gather_dict_list, CYZD_text)

    def infer_tgi_server(client, inputs):
        answer = "姓名：" + '\n'
        answer += patient_name + '\n\n\n'
        answer = "性别：" + '\n'
        answer += sex + '\n\n\n'
        answer = "年龄：" + '\n'
        answer += str(age) + '\n\n\n'

        answer = "入院诊断：" + '\n'
        answer += RYZD_text + '\n\n\n'

        answer += "出院诊断：" + '\n'
        answer += CYZD_text + '\n\n\n'

        answer += "住院天数：" + '\n'
        answer += str(length_of_stay) + '\n\n\n'

        answer += "入院情况：" + '\n'
        answer += SCBCJL_ZDYJ + '\n\n\n'

        answer += "住院经过：" + '\n'
        # 还要改
        answer += "住院经过(检查结果)：" + '\n'
        answer += hospitalization_check_text + '\n\n\n'

        answer += "住院经过(检验结果)：" + '\n'
        answer += hospitalization_test_text + '\n\n\n'

        # yield json.dumps(answer, ensure_ascii=False) + '\n'

        # tgi前内容输出
        for a in answer:
            yield json.dumps(a, ensure_ascii=False) + '\n'
            # yield a + '\n'

        # 调用tgi的api进行推理
        first = True
        tgi_answer = ""
        for output in client.generate_stream(inputs["inputs"], **inputs["parameters"]):
            if not output.token.special:
                if first:
                    new_text = output.token.text.lstrip()
                    first = False
                else:
                    new_text = output.token.text
                tgi_answer += new_text
                answer += new_text

                # yield new_text
                yield json.dumps(new_text, ensure_ascii=False) + '\n'
                # yield new_text + '\n'

        # tgi后内容输出
        after_tgi_answer = "\n\n\n"

        after_tgi_answer += "出院情况：" + '\n'
        after_tgi_answer += discharge_status + '\n\n\n'

        after_tgi_answer += "出院医嘱：" + '\n'
        after_tgi_answer += order_of_discharge_text + '\n\n\n'

        after_tgi_answer += "出院去向：" + '\n'
        after_tgi_answer += "回家"

        answer += after_tgi_answer

        # yield json.dumps(after_tgi_answer, ensure_ascii=False) + '\n'

        for aa in after_tgi_answer:
            yield json.dumps(aa, ensure_ascii=False) + '\n'
            # yield aa + '\n'
        print(answer)

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }

    return Response(infer_tgi_server(client, inputs), mimetype="text/event-stream", headers=headers)


"""
gunicorn -b 0.0.0.0:9600 -t 100 --log-level=debug flask_infer:app


curl --location --request POST 'http://gbox11.aigauss.com:9601/infer' \
--header 'Content-Type: application/json' \
--data-raw '{
    "discharge_id": "xxx",
"data":"xxx",
"sex":"xxx",
"mrn":"xxx",
"series":"xxx",
"admission_date":"xxx",
"discharge_data":"xxx",

"ZZYSCFJL":[{"DOC_TIME":{
"key":"DOC_TIME",
"name":"DOC_TIME",
"value":"xxx"
},
"CFJL":{
"key":"CFJL",
"name":"查房记录",
"value":"xxx"
},
"BRZS": {
"key":"BRZS",
"name":"病人主诉",
"value":"xxx"
},
"CT":{
"key":"CT",
"name":"查体",
"value":"xxx"
},
"JCJYJG":{
"key":"JCJYJG",
"name":"检查检验结果",
"value":"xxx"
},
"BCZD":{
"key":"BCZD",
"name":"病程诊断",
"value":"xxx"
},
"BQFXYZLJH":{
"key":"BQFXYZLJH",
"name":"病情分析与诊疗计划",
"value":"xxx"
}},
{"DOC_TIME":{
"key":"DOC_TIME",
"name":"DOC_TIME",
"value":"xxx"
},
"CFJL":{
"key":"CFJL",
"name":"查房记录",
"value":"xxx"
},
"BRZS": {
"key":"BRZS",
"name":"病人主诉",
"value":"xxx"
},
"CT":{
"key":"CT",
"name":"查体",
"value":"xxx"
},
"JCJYJG":{
"key":"JCJYJG",
"name":"检查检验结果",
"value":"xxx"
},
"BCZD":{
"key":"BCZD",
"name":"病程诊断",
"value":"xxx"
},
"BQFXYZLJH":{
"key":"BQFXYZLJH",
"name":"病情分析与诊疗计划",
"value":"xxx"
}}

],


"ICULHCFJL":[{"DOC_TIME":{
"key":"DOC_TIME",
"name":"DOC_TIME",
"value":"xxx"
},
"CFJL":{
"key":"CFJL",
"name":"查房记录",
"value":"xxx"
},
"BRZS": {
"key":"BRZS",
"name":"病人主诉",
"value":"xxx"
},
"CT":{
"key":"CT",
"name":"查体",
"value":"xxx"
},
"JCJYJG":{
"key":"JCJYJG",
"name":"检查检验结果",
"value":"xxx"
},
"BCZD":{
"key":"BCZD",
"name":"病程诊断",
"value":"xxx"
},
"BQFXYZLJH":{
"key":"BQFXYZLJH",
"name":"病情分析与诊疗计划",
"value":"xxx"
}}],
"ZRYSCFJL":[{"DOC_TIME":{
"key":"DOC_TIME",
"name":"DOC_TIME",
"value":"xxx"
},
"CFJL":{
"key":"CFJL",
"name":"查房记录",
"value":"xxx"
},
"BRZS": {
"key":"BRZS",
"name":"病人主诉",
"value":"xxx"
},
"CT":{
"key":"CT",
"name":"查体",
"value":"xxx"
},
"JCJYJG":{
"key":"JCJYJG",
"name":"检查检验结果",
"value":"xxx"
},
"BCZD":{
"key":"BCZD",
"name":"病程诊断",
"value":"xxx"
},
"BQFXYZLJH":{
"key":"BQFXYZLJH",
"name":"病情分析与诊疗计划",
"value":"xxx"
}}],

"KZRCFJL":[{"DOC_TIME":{
"key":"DOC_TIME",
"name":"DOC_TIME",
"value":"xxx"
},
"CFJL":{
"key":"CFJL",
"name":"查房记录",
"value":"xxx"
},
"BRZS": {
"key":"BRZS",
"name":"病人主诉",
"value":"xxx"
},
"CT":{
"key":"CT",
"name":"查体",
"value":"xxx"
},
"JCJYJG":{
"key":"JCJYJG",
"name":"检查检验结果",
"value":"xxx"
},
"BCZD":{
"key":"BCZD",
"name":"病程诊断",
"value":"xxx"
},
"BQFXYZLJH":{
"key":"BQFXYZLJH",
"name":"病情分析与诊疗计划",
"value":"xxx"
}}],


"RCBCJL":
[{"DOC_TIME":{
"key":"DOC_TIME",
"name":"DOC_TIME",
"value":"xxx"
},
"CFJL":{
"key":"CFJL",
"name":"查房记录",
"value":"xxx"
},
"BRZS": {
"key":"BRZS",
"name":"病人主诉",
"value":"xxx"
},
"CT":{
"key":"CT",
"name":"查体",
"value":"xxx"
},
"JCJYJG":{
"key":"JCJYJG",
"name":"检查检验结果",
"value":"xxx"
},
"BCZD":{
"key":"BCZD",
"name":"病程诊断",
"value":"xxx"
},
"BQFXYZLJH":{
"key":"BQFXYZLJH",
"name":"病情分析与诊疗计划",
"value":"xxx"
}}],


"SCBRJL":{"ZDYJ":
{
"key":"ZDYJ",
"name":"诊断依据",
"value":"xxx"
}},

"z4_interface_output":

{"RYZD_interface":
{
"key":"RYZD_interface",
"name":"入院诊断_接口",
"value":"xxx"
},
"CYZD_interface":
{
"key":"CYZD_interface",
"name":"出院诊断_接口",
"value":"xxx"
},
"JC_interface":
{
"key":"JC_interface",
"name":"检查_接口",
"value":"xxx"
},
"JY_interface":{
"key":"JY_interface",
"name":"检验_接口",
"value":"xxx"
},
"CYYZ_interface":{
"key":"CYYZ_interface",
"name":"出院医嘱_接口",
"value":"xxx"
}
}
}'


curl --location --request POST 'http://gbox11.aigauss.com:9600/infer' \
--header 'Content-Type: application/json' \
--data-raw '{
"discharge_id":"xxx",
"data":"xxx",
"sex":"xxx",
"mrn":"xxx",
"series":"xxx",
"admission_date":"xxx",
"discharge_data":"xxx",
    "z4__interface_output":{
        "JC":[{"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:26:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "心电图检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "常规十二导心电图检测", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "1.窦性心律\n \n2.房性早搏\n \n3.室性早搏", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "1.窦性心律\n \n2.房性早搏\n \n3.室性早搏", "name": "检查所见"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:26:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "B超检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "(VTE)右下肢动静脉超声检查", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "US1860997", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "US1860997", "name": "检查所见"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:26:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "B超检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "(VTE)左下肢动静脉超声检查", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "US1860997", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "US1860997", "name": "检查所见"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:26:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "B超检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "恶性肿瘤浅表彩超检查（双侧颌下、颈部、锁骨上、腋下、腹股沟）", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "US1860997", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "US1860997", "name": "检查所见"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:27:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "B超检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "盆腔淋巴结彩超检查", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "US1860997", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "US1860997", "name": "检查所见"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:30:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "B超检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "肝,胆,胰,脾彩超检查", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "1. 胆囊壁毛糙\n2. 右肾囊肿 双肾多发结晶\n3. 右颈部一淋巴结形态稍饱满\n 双侧锁骨上一淋巴结形态稍饱满\n 左颈部、双侧颌下、双腋下、双侧腹股沟、盆腔均未见明显肿大淋巴结\n4. 后腹膜扫查未见明显异常\n5. 双侧肾上腺超声未见明显异常\n6. 双下肢动脉内膜毛糙\n 双下肢深静脉血流通畅", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "1. 胆囊壁毛糙\n2. 右肾囊肿 双肾多发结晶\n3. 右颈部一淋巴结形态稍饱满\n 双侧锁骨上一淋巴结形态稍饱满\n 左颈部、双侧颌下、双腋下、双侧腹股沟、盆腔均未见明显肿大淋巴结\n4. 后腹膜扫查未见明显异常\n5. 双侧肾上腺超声未见明显异常\n6. 双下肢动脉内膜毛糙\n 双下肢深静脉血流通畅", "name": "检查所见"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:30:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "B超检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "女泌尿系统(肾,输尿管,膀胱)彩超检查", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "US1860997", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "US1860997", "name": "检查所见"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:30:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "B超检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "后腹膜彩超检查", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "US1860997", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "US1860997", "name": "检查所见"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:30:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "B超检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "双侧肾上腺彩超检查", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "US1860997", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "US1860997", "name": "检查所见"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:30:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "CT检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "胸部CT平扫", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "左肺下叶肺癌治疗后复查，较2024-03-10CT左肺下叶病灶稍减小，请随诊。\n右肺下叶结片影伴空泡形成，两肺多发磨玻璃结节，较前大致相仿，建议复查。\n两肺散在结节，部分较前新发，转移瘤待排，请结合临床\n对照前片，右肺上叶实性结节较前显示欠清；右上肺血管旁实性小结节，大致同前，请随诊。\n两肺散在纤维增殖钙化灶；慢性支气管炎症、两肺局限性肺气肿。\n冠状动脉、主动脉粥样硬化。\n食管下段旁、腹膜后增大淋巴结影，转移性考虑，请结合临床。\n附见：甲状腺两侧叶密度不均匀，建议结合超声检查。胸椎退行性改变，胸10、11椎体高密度影，较前片增多，转移考虑，请随诊。两侧部分肋骨骨皮质欠规整，右侧第9肋骨、右侧第6肋骨病理性骨折考虑，局部稍高密度影，转移待排。肝脏低密度影。脾脏钙化灶。", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "左肺下叶肺癌治疗后复查，较2024-03-10CT左肺下叶病灶稍减小，请随诊。\n右肺下叶结片影伴空泡形成，两肺多发磨玻璃结节，较前大致相仿，建议复查。\n两肺散在结节，部分较前新发，转移瘤待排，请结合临床\n对照前片，右肺上叶实性结节较前显示欠清；右上肺血管旁实性小结节，大致同前，请随诊。\n两肺散在纤维增殖钙化灶；慢性支气管炎症、两肺局限性肺气肿。\n冠状动脉、主动脉粥样硬化。\n食管下段旁、腹膜后增大淋巴结影，转移性考虑，请结合临床。\n附见：甲状腺两侧叶密度不均匀，建议结合超声检查。胸椎退行性改变，胸10、11椎体高密度影，较前片增多，转移考虑，请随诊。两侧部分肋骨骨皮质欠规整，右侧第9肋骨、右侧第6肋骨病理性骨折考虑，局部稍高密度影，转移待排。肝脏低密度影。脾脏钙化灶。", "name": "检查所见"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "rq": {"key": "rq", "value": "2024-05-08 15:31:00", "name": "日期"}, "jclb": {"key": "jclb", "value": "MRI检查申请单", "name": "检查类别"}, "jcxm": {"key": "jcxm", "value": "颅脑MRI平扫+T2flair+DWI", "name": "检查项目"}, "zdyx": {"key": "zdyx", "value": "右侧额叶新近脑梗死。\n脑干、两侧额顶叶皮层下、侧脑室旁散在缺血性改变。\n脑干及小脑半球小软化灶可能。\n脑萎缩。\n附见：右侧乳突少许炎症。部分鼻窦粘膜增厚。C3椎体信号不均匀。", "name": "诊断影像"}, "jcsj": {"key": "jcsj", "value": "右侧额叶新近脑梗死。\n脑干、两侧额顶叶皮层下、侧脑室旁散在缺血性改变。\n脑干及小脑半球小软化灶可能。\n脑萎缩。\n附见：右侧乳突少许炎症。部分鼻窦粘膜增厚。C3椎体信号不均匀。", "name": "检查所见"}}]
,"JY":[{"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "随机血葡萄糖", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "4.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;11.1", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "肌酸激酶", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "258", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "40-200", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "肌酸激酶－MB(酶活性)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "11", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;25", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "乳酸脱氢酶", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "208", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "120-250", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "总胆红素", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "8.3", "name": "项目结果"}, "dw": {"key": "dw", "value": "μmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;21.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "直接胆红素", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.3", "name": "项目结果"}, "dw": {"key": "dw", "value": "μmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;6.8", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "氯(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "115", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "99-110", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿素(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "6.84", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.1-8.8", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "肌酐(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "59", "name": "项目结果"}, "dw": {"key": "dw", "value": "μmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "41-81", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血淀粉酶(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "107.9", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "35-135", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "丙氨酸氨基转移酶(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "10", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "7-45", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "总钙(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.97", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "2.11-2.52", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "胆碱脂酶", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "7.06", "name": "项目结果"}, "dw": {"key": "dw", "value": "KU/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "5.59-12.28", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "脂肪酶", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "60.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1-60", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿酸", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "223.2", "name": "项目结果"}, "dw": {"key": "dw", "value": "μmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0-416", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "C反应蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.7", "name": "项目结果"}, "dw": {"key": "dw", "value": "mg/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;6", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "钾(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "4.08", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.5-5.3", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "钠(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "140.6", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "137-147", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "间接胆红素", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "6", "name": "项目结果"}, "dw": {"key": "dw", "value": "μmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;19.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "总蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "61.2", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "65-85", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "白蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "36.6", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "40-55", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "球蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "24.6", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "20-40", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "白蛋白/球蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1.2-2.4", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "急诊生化全套+心肌酶谱", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "天门冬氨酸氨基转移酶", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "18", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "13-40", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "白细胞计数", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "4.4", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.5-9.5", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "中性粒细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "62", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "40.0-75.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "淋巴细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "24.9", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "20.0-50.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "单核细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "10.6", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.0-10.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜酸性粒细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.9", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.4-8.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜碱性粒细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.6", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.0-1.5", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板计数", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "227", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "125-350", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板压积", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.18", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.106-0.250", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板平均体积", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "8", "name": "项目结果"}, "dw": {"key": "dw", "value": "fl", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "7.8-11.3", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板体积分布宽度", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "15.6", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "13.0-18.5", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血红蛋白测定", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "100", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "115-150", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "红细胞比积测定", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "28.9", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "35.0-45.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "平均红细胞体积测定", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "96.9", "name": "项目结果"}, "dw": {"key": "dw", "value": "fl", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "84.0-100.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "平均红细胞血红蛋白量", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "33.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "pg", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "27.0-34.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "平均红细胞血红蛋白浓度", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "346", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "316-354", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "红细胞体积分布宽度", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "20.2", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "12.0-15.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "中性粒细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.7", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1.8-6.3", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "淋巴细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.1", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1.1-3.2", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "单核细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.1-0.6", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜酸性粒细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.08", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.02-0.52", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜碱性粒细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.03", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0-0.10", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "红细胞计数", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.98", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^12/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.8-5.1", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "B型纳尿肽定量测定(BNP)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "B型尿钠肽", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "20.2", "name": "项目结果"}, "dw": {"key": "dw", "value": "pg/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;100", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肌钙蛋白(急)(首诊)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "高敏肌钙蛋白T", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.009", "name": "项目结果"}, "dw": {"key": "dw", "value": "ng/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;0.014", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "D二聚体(急)/凝血功能常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "凝血酶原时间", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "11.3", "name": "项目结果"}, "dw": {"key": "dw", "value": "s", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "9.8-12.3", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "D二聚体(急)/凝血功能常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "国际标准化比值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.90-1.10", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "D二聚体(急)/凝血功能常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "活化部分凝血活酶时间", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "25.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "s", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "23.9-33.5", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "D二聚体(急)/凝血功能常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "凝血酶时间", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "17.1", "name": "项目结果"}, "dw": {"key": "dw", "value": "s", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "14.0-21.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "D二聚体(急)/凝血功能常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "纤维蛋白原", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.61", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1.80-3.50", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "D二聚体(急)/凝血功能常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "D-二聚体", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.14", "name": "项目结果"}, "dw": {"key": "dw", "value": "mg/L FEU", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;0.50", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "颜色", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "稻黄色", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "浊度", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "清亮", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿比重", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.018", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1.003-1.030", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿pH", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "6", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "4.6-8.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿潜血", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "±", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "阴性", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "白细胞酯酶", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "阴性", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "阴性", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "红细胞", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "10.4", "name": "项目结果"}, "dw": {"key": "dw", "value": "/μL", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0-25", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "白细胞", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.4", "name": "项目结果"}, "dw": {"key": "dw", "value": "/μL", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0-25", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "上皮细胞", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "6.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "/μL", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0-20", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "管型", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.27", "name": "项目结果"}, "dw": {"key": "dw", "value": "/μL", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0-2", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿蛋白质", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "阴性", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "阴性", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿葡萄糖", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "阴性", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "阴性", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿胆原", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "正常", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "正常", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿胆红素", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "阴性", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "阴性", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿酮体", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "阴性", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "阴性", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "尿液分析(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿亚硝酸盐", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "阴性", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "阴性", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "粪便常规", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "粪便颜色", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "黄色", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "粪便常规", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "粪便性状", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "软便", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "粪便常规", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "粪便红细胞", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "未见", "name": "项目结果"}, "dw": {"key": "dw", "value": "/HP", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "未见", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "粪便常规", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "粪便白细胞", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "未见", "name": "项目结果"}, "dw": {"key": "dw", "value": "/HP", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0-2", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "粪便常规", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "真菌", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "未见", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "未见", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "粪便隐血试验", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "粪便隐血试验", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "阴性", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "阴性", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肿瘤标志物（女）", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "癌胚抗原", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "26.44", "name": "项目结果"}, "dw": {"key": "dw", "value": "ng/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;5.00", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肿瘤标志物（女）", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "甲胎蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.82", "name": "项目结果"}, "dw": {"key": "dw", "value": "ng/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;7.00", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肿瘤标志物（女）", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "神经元特异烯醇化酶", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "12.94", "name": "项目结果"}, "dw": {"key": "dw", "value": "ng/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;20.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肿瘤标志物（女）", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "细胞角蛋白21-1", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.24", "name": "项目结果"}, "dw": {"key": "dw", "value": "ng/mL", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;2.08", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肿瘤标志物（女）", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "糖链抗原19-9", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "6.8", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;43.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肿瘤标志物（女）", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "糖链抗原24-2", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "11.72", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;25.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肿瘤标志物（女）", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "糖链抗原125", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "14.8", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/mL", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;22.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肿瘤标志物（女）", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "糖链抗原72-4", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "7.91", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;10.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肿瘤标志物（女）", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "鳞状细胞癌相关抗原", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.1", "name": "项目结果"}, "dw": {"key": "dw", "value": "ng/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;1.50", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "肿瘤标志物（女）", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "糖链抗原15-3", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "9.2", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:25:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;31.3", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "胃泌素释放前肽", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "胃泌素释放肽前体", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "790.34", "name": "项目结果"}, "dw": {"key": "dw", "value": "pg/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-08 15:26:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;65.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "乙肝病毒DNA检测", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "乙型肝炎病毒DNA", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "&lt;1.00×10^2", "name": "项目结果"}, "dw": {"key": "dw", "value": "IU/ml", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:17:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "15", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "白细胞计数", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "3.6", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.5-9.5", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "中性粒细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "64.9", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "40.0-75.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "淋巴细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "27.9", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "20.0-50.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "单核细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "4.8", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.0-10.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜酸性粒细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.8", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.4-8.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜碱性粒细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.6", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.0-1.5", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "中性粒细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.3", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1.8-6.3", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "淋巴细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1.1-3.2", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "单核细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.2", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.1-0.6", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜酸性粒细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.06", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.02-0.52", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜碱性粒细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.02", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0-0.10", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "红细胞计数", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.87", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^12/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.8-5.1", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血红蛋白测定", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "96", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "115-150", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "红细胞比积测定", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "28.1", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "35.0-45.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "平均红细胞体积测定", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "97.9", "name": "项目结果"}, "dw": {"key": "dw", "value": "fl", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "84.0-100.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "平均红细胞血红蛋白量", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "33.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "pg", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "27.0-34.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "平均红细胞血红蛋白浓度", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "342", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "316-354", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "红细胞体积分布宽度", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "19.6", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "12.0-15.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板计数", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "205", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "125-350", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板压积", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.19", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.106-0.250", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板平均体积", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "9.3", "name": "项目结果"}, "dw": {"key": "dw", "value": "fl", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "7.8-11.3", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板体积分布宽度", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "15.7", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-10 14:18:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "13.0-18.5", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "总胆红素", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "8.4", "name": "项目结果"}, "dw": {"key": "dw", "value": "μmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;21.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "直接胆红素", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.4", "name": "项目结果"}, "dw": {"key": "dw", "value": "μmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;6.8", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "间接胆红素", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "6", "name": "项目结果"}, "dw": {"key": "dw", "value": "μmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;19.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "总蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "58.6", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "65-85", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "白蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "35.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "40-55", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "球蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "23.1", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "20-40", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "白蛋白/球蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1.2-2.4", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "天门冬氨酸氨基转移酶", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "16", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "13-40", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "丙氨酸氨基转移酶(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "9", "name": "项目结果"}, "dw": {"key": "dw", "value": "U/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "7-45", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿酸", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "202.3", "name": "项目结果"}, "dw": {"key": "dw", "value": "μmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0-416", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "尿素(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "9.72", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.1-8.8", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "肌酐(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "59", "name": "项目结果"}, "dw": {"key": "dw", "value": "μmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "41-81", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "钾(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "4.32", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.5-5.3", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "钠(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "139", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "137-147", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "氯(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "112.1", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "99-110", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "电解质四项(急)/肾功能(急)/肝功能(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "总钙(急)", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.03", "name": "项目结果"}, "dw": {"key": "dw", "value": "mmol/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "2.11-2.52", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "白细胞计数", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "3.3", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.5-9.5", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "中性粒细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "65.5", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "40.0-75.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "淋巴细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "29.8", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "20.0-50.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "单核细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1.8", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.0-10.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜酸性粒细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.2", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.4-8.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜碱性粒细胞百分比", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.7", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.0-1.5", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "中性粒细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.2", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1.8-6.3", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "淋巴细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "1", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "1.1-3.2", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "单核细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.1", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.1-0.6", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜酸性粒细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.07", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.02-0.52", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "嗜碱性粒细胞绝对值", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.02", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0-0.10", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "红细胞计数", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "2.77", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^12/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "3.8-5.1", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血红蛋白测定", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "94", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "115-150", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "红细胞比积测定", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "27.2", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "L", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "35.0-45.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "平均红细胞体积测定", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "98.3", "name": "项目结果"}, "dw": {"key": "dw", "value": "fl", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "84.0-100.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "平均红细胞血红蛋白量", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "33.8", "name": "项目结果"}, "dw": {"key": "dw", "value": "pg", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "27.0-34.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "平均红细胞血红蛋白浓度", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "346", "name": "项目结果"}, "dw": {"key": "dw", "value": "g/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "316-354", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "红细胞体积分布宽度", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "19.1", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "H", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "12.0-15.0", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板计数", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "185", "name": "项目结果"}, "dw": {"key": "dw", "value": "×10^9/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "125-350", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板压积", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "0.18", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "0.106-0.250", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板平均体积", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "9.8", "name": "项目结果"}, "dw": {"key": "dw", "value": "fl", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "7.8-11.3", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "血小板体积分布宽度", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "15.8", "name": "项目结果"}, "dw": {"key": "dw", "value": "%", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "13.0-18.5", "name": "参考范围"}}, {"mrn": {"key": "mrn", "value": "98872", "name": "MRN"}, "series": {"key": "series", "value": "193", "name": "SERIES"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "xb": {"key": "xb", "value": "F", "name": "性别"}, "ks": {"key": "ks", "value": "呼吸与危重症医学科", "name": "科室"}, "jcmc": {"key": "jcmc", "value": "血常规+CRP(全血)(急)", "name": "检查名称"}, "xmmc": {"key": "xmmc", "value": "C反应蛋白", "name": "项目名称"}, "xmjg": {"key": "xmjg", "value": "&lt;0.20", "name": "项目结果"}, "dw": {"key": "dw", "value": "mg/L", "name": "单位"}, "sfyc": {"key": "sfyc", "value": "n", "name": "是否异常"}, "sj": {"key": "sj", "value": "2024-05-14 12:00:00", "name": "时间"}, "ckfw": {"key": "ckfw", "value": "&lt;6.0", "name": "参考范围"}}]
, "CYZD": [
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "恶性肿瘤的维持性化疗"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "1"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "小细胞肺癌(广泛期 cTxNxM1)"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "K"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "恶性肿瘤靶向治疗  肺腺癌，cTXN3M1c，IVB期，EGFR 21外显子L858R突变"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "K"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "肺部肿瘤消融术后"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "高血压"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "肝囊肿"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "K"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "肾囊肿"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "K"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "动脉硬化 主动脉硬化 双下肢斑块形成"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "K"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "便秘 内痔"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "结肠腺瘤EMR术后"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "K"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "缺血性肠病考虑"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20T02:37:26.000Z"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "腰椎间盘突出  颈椎间盘突出"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "甲状腺结节"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "轻度贫血"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "膝关节退行性骨关节病"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "睡眠障碍"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "房性期前收缩[房性早搏]"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "室性期前收缩"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    },
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": 98872
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": 193
        },
        "zd": {
            "key": "zd",
            "name": "诊断",
            "value": "脑梗死"
        },
        "zdrq": {
            "key": "zdrq",
            "name": "诊断日期",
            "value": "2024-05-20 02:37:26.000"
        },
        "zdbz": {
            "key": "zdbz",
            "name": "诊断标志",
            "value": "H"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "cyzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 02:37:26.000"
        }
    }
],
"RYZD": [
    {
        "blh": {
            "key": "blh",
            "name": "病历号",
            "value": "98872"
        },
        "jzcs": {
            "key": "jzcs",
            "name": "就诊次数",
            "value": "193"
        },
        "cjsj": {
            "key": "cjsj",
            "name": "创建时间",
            "value": "2024-05-08 07:37:12.000"
        },
        "gxsj": {
            "key": "gxsj",
            "name": "更新时间",
            "value": "2024-05-20 03:08:30.000"
        },
        "blmc": {
            "key": "blmc",
            "name": "病历名称",
            "value": "入院记录(大病历新)"
        },
        "bizType": {
            "key": "bizType",
            "name": "bizType",
            "value": "ryzd"
        },
        "doc_time": {
            "key": "doc_time",
            "name": "doc_time",
            "value": "2024-05-20 03:08:30.000"
        },
        "病历ID": {
            "key": "病历ID",
            "name": "病历ID",
            "value": "98872_34097-0214_20240508153627_2427"
        },
        "cbzd": {
            "key": "cbzd",
            "name": "初步诊断",
            "value": null
        }
    }
],"CYYZ":[{"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "★骨化三醇软胶囊", "name": "药品"}, "gg": {"key": "gg", "value": "0.25ug", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每日一次", "name": "频度"}, "sl": {"key": "sl", "value": "1.00盒", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "★碳酸钙D3片", "name": "药品"}, "gg": {"key": "gg", "value": "600mg/125IU", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每日一次", "name": "频度"}, "sl": {"key": "sl", "value": "1.00瓶", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "(危)阿普唑仑片", "name": "药品"}, "gg": {"key": "gg", "value": "0.4mg", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每晚一次", "name": "频度"}, "sl": {"key": "sl", "value": "0.35片", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "★苯磺酸氨氯地平片", "name": "药品"}, "gg": {"key": "gg", "value": "5mg", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每日一次", "name": "频度"}, "sl": {"key": "sl", "value": "1.00盒", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "★(20mg)普伐他汀钠片", "name": "药品"}, "gg": {"key": "gg", "value": "20mg", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每晚一次", "name": "频度"}, "sl": {"key": "sl", "value": "4.00盒", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "★复方消化酶胶囊", "name": "药品"}, "gg": {"key": "gg", "value": "复方", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每日三次", "name": "频度"}, "sl": {"key": "sl", "value": "2.00盒", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "醋酸甲地孕酮分散片", "name": "药品"}, "gg": {"key": "gg", "value": "160mg", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每日一次", "name": "频度"}, "sl": {"key": "sl", "value": "2.00盒", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "盐酸伊托必利片", "name": "药品"}, "gg": {"key": "gg", "value": "50mg", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每日三次", "name": "频度"}, "sl": {"key": "sl", "value": "2.00盒", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "利可君片", "name": "药品"}, "gg": {"key": "gg", "value": "20mg", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每日三次", "name": "频度"}, "sl": {"key": "sl", "value": "1.00盒", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "★硫酸氢氯吡格雷片", "name": "药品"}, "gg": {"key": "gg", "value": "75mg", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每日一次", "name": "频度"}, "sl": {"key": "sl", "value": "1.00盒", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}, {"blh": {"key": "blh", "value": "98872", "name": "病历号"}, "jzcs": {"key": "jzcs", "value": "193", "name": "就诊次数"}, "xm": {"key": "xm", "value": "鲍云秀", "name": "姓名"}, "yp": {"key": "yp", "value": "★硫酸氢氯吡格雷片", "name": "药品"}, "gg": {"key": "gg", "value": "75mg", "name": "规格"}, "yf": {"key": "yf", "value": "口服", "name": "用法"}, "pd": {"key": "pd", "value": "每日一次", "name": "频度"}, "sl": {"key": "sl", "value": "2.00盒", "name": "数量"}, "yzlx": {"key": "yzlx", "value": "出院带药", "name": "医嘱类型"}}]

},
    "SCBCJL":[],
    "ZZYSCFJL": [
        {
            "blh": {
                "key": "blh",
                "name": "病历号",
                "value": "98872"
            },
            "jzcs": {
                "key": "jzcs",
                "name": "就诊次数",
                "value": "193"
            },
            "cjsj": {
                "key": "cjsj",
                "name": "创建时间",
                "value": "2024-05-17 08:16:01.000"
            },
            "gxsj": {
                "key": "gxsj",
                "name": "更新时间",
                "value": "2024-05-17 08:47:42.000"
            },
            "blmc": {
                "key": "blmc",
                "name": "病历名称",
                "value": "主治医师查房记录"
            },
            "病历ID": {
                "key": "病历ID",
                "name": "病历ID",
                "value": "98872_34130-0412_20240517161554_6041"
            },
            "brzs": {
                "key": "brzs",
                "name": "病人主诉",
                "value": "无明显不适"
            },
            "bizType": {
                "key": "bizType",
                "name": "bizType",
                "value": "zzyscfjl"
            },
            "doc_time": {
                "key": "doc_time",
                "name": "doc_time",
                "value": "2024-05-17 08:45:44.000"
            },
            "ct": {
                "key": "ct",
                "name": "查体",
                "value": "神清，精神可。皮肤巩膜无黄染。双侧颈部、锁骨上淋巴结未触及肿大。颈软，胸廓无畸形，两肺呼吸音清，未闻及明显干湿性啰音。心律齐，心脏各瓣膜区未闻及病理性杂音。腹软，无压痛及反跳痛，肝脾肋下未触及。双下肢无水肿。四肢肌力正常，双侧巴氏征阴性。"
            },
            "jcjyjg": {
                "key": "jcjyjg",
                "name": "检查检验结果",
                "value": "(2024-05-08)胸部CT平扫示：左肺下叶肺癌治疗后复查，较2024-03-10CT左肺下叶病灶稍减小，请随诊。右肺下叶结片影伴空泡形成，两肺多发磨玻璃结节，较前大致相仿，建议复查。两肺散在结节，部分较前新发，转移瘤待排，请结合临床；对照前片，右肺上叶实性结节较前显示欠清；右上肺血管旁实性小结节，大致同前，请随诊。两肺散在纤维增殖钙化灶；慢性支气管炎症、两肺局限性肺气肿。冠状动脉、主动脉粥样硬化。食管下段旁、腹膜后增大淋巴结影，转移性考虑，请结合临床。附见：甲状腺两侧叶密度不均匀，建议结合超声检查。胸椎退行性改变，胸10、11椎体高密度影，较前片增多，转移考虑，请随诊。两侧部分肋骨骨皮质欠规整，右侧第9肋骨、右侧第6肋骨病理性骨折考虑，局部稍高密度影，转移待排。肝脏低密度影。脾脏钙化灶。(2024-05-09)常规十二导心电图提示：1.窦性心律；2.房性早搏；3.室性早搏。(2024-05-09) 彩超示：1. 胆囊壁毛糙；2. 右肾囊肿 双肾多发结晶；3. 右颈部一淋巴结形态稍饱满； 双侧锁骨上一淋巴结形态稍饱满； 左颈部、双侧颌下、双腋下、双侧腹股沟、盆腔均未见明显肿大淋巴结；4. 后腹膜扫查未见明显异常；5.双侧肾上腺超声未见明显异常；6. 双下肢动脉内膜毛糙； 双下肢深静脉血流通畅。"
            },
            "zd": {
                "key": "zd",
                "name": "诊断",
                "value": "1.小细胞肺癌（广泛期 cTxNxM1） 2.恶性肿瘤靶向治疗 肺腺癌，cTxN3M1c IVB期，EGFR21外显子L858R点突变 肺部肿瘤消融术后 恶性肿瘤化学治疗后 2.高血压 3.肝囊肿 4.肾囊肿 5.动脉粥样硬化 主动脉硬化 双下肢动脉斑块形成  6.便秘 内痔 7.结肠腺瘤EMR术后 缺血性肠病考虑 8.腰椎间盘突出 颈椎间盘突出 9.甲状腺结节 10.轻度贫血 11.膝关节退行性骨关节病 12.睡眠障碍  13.房性早搏 室性早搏  "
            },
            "bqfxyzljh": {
                "key": "bqfxyzljh",
                "name": "病情分析与诊疗计划",
                "value": "患者老年女性，基础有高血压、甲状腺结节、肝肾囊肿等病史，无过敏史，慢性病程；此次因“确诊肺腺癌5年余，确诊小细胞肺癌2月”入院；目前阿美替尼110mg qd靶向治疗中；查体：两肺未闻及明显干湿啰音。拟完善相关检查排除禁忌后行下一周期抗肿瘤治疗，余治疗继续予以阿美替尼抗肿瘤治疗，苯磺酸氨氯地平片降压，阿托伐他汀钙片降脂稳斑，甲钴胺营养神经，护胃，助消化等对症治疗。继观。  "
            }
        },
        {
            "blh": {
                "key": "blh",
                "name": "病历号",
                "value": "98872"
            },
            "jzcs": {
                "key": "jzcs",
                "name": "就诊次数",
                "value": "193"
            },
            "cjsj": {
                "key": "cjsj",
                "name": "创建时间",
                "value": "2024-05-17 08:16:23.000"
            },
            "gxsj": {
                "key": "gxsj",
                "name": "更新时间",
                "value": "2024-05-17 08:47:45.000"
            },
            "blmc": {
                "key": "blmc",
                "name": "病历名称",
                "value": "主治医师查房记录"
            },
            "病历ID": {
                "key": "病历ID",
                "name": "病历ID",
                "value": "98872_34130-0412_20240517161615_8327"
            },
            "brzs": {
                "key": "brzs",
                "name": "病人主诉",
                "value": "用药后无恶心呕吐等不适，稍感头晕"
            },
            "bizType": {
                "key": "bizType",
                "name": "bizType",
                "value": "zzyscfjl"
            },
            "doc_time": {
                "key": "doc_time",
                "name": "doc_time",
                "value": "2024-05-17 08:45:44.000"
            },
            "ct": {
                "key": "ct",
                "name": "查体",
                "value": " 神清，精神可。皮肤巩膜无黄染。双侧颈部、锁骨上淋巴结未触及肿大。颈软，胸廓无畸形，两肺呼吸音清，未闻及明显干湿性啰音。心律齐，心脏各瓣膜区未闻及病理性杂音。腹软，无压痛及反跳痛，肝脾肋下未触及。双下肢无水肿。四肢肌力正常，双侧巴氏征阴性。"
            },
            "jcjyjg": {
                "key": "jcjyjg",
                "name": "检查检验结果",
                "value": "暂无更新"
            },
            "zd": {
                "key": "zd",
                "name": "诊断",
                "value": "1.小细胞肺癌（广泛期 cTxNxM1） 2.恶性肿瘤靶向治疗 肺腺癌，cTxN3M1c IVB期，EGFR21外显子L858R点突变 肺部肿瘤消融术后 恶性肿瘤化学治疗后 2.高血压 3.肝囊肿 4.肾囊肿 5.动脉粥样硬化 主动脉硬化 双下肢动脉斑块形成  6.便秘 内痔 7.结肠腺瘤EMR术后 缺血性肠病考虑 8.腰椎间盘突出 颈椎间盘突出 9.甲状腺结节 10.轻度贫血 11.膝关节退行性骨关节病 12.睡眠障碍  13.房性早搏 室性早搏 "
            },
            "bqfxyzljh": {
                "key": "bqfxyzljh",
                "name": "病情分析与诊疗计划",
                "value": "患者者老年女性，基础有高血压、甲状腺结节、肝肾囊肿等病史，慢性病程；此次因“确诊肺腺癌5年余，确诊小细胞肺癌2月”入院；患者肺腺癌控制稳定，继续阿美替尼110mg qd。昨日予卡铂350mg+依托泊苷0.12g治疗，患者诉稍感头晕，可自行缓解。现一般状况可，继续D2化疗。"
            }
        },
        {
            "bizType": {
                "key": "bizType",
                "name": "bizType",
                "value": "zzyscfjl"
            },
            "doc_time": {
                "key": "doc_time",
                "name": "doc_time",
                "value": "2024-05-17 08:45:44.000"
            },
            "blh": {
                "key": "blh",
                "name": "病历号",
                "value": "98872"
            },
            "jzcs": {
                "key": "jzcs",
                "name": "就诊次数",
                "value": "193"
            },
            "cjsj": {
                "key": "cjsj",
                "name": "创建时间",
                "value": "2024-05-17 08:16:36.000"
            },
            "gxsj": {
                "key": "gxsj",
                "name": "更新时间",
                "value": "2024-05-17 08:47:48.000"
            },
            "blmc": {
                "key": "blmc",
                "name": "病历名称",
                "value": "主治医师查房记录"
            },
            "病历ID": {
                "key": "病历ID",
                "name": "病历ID",
                "value": "98872_34130-0412_20240517161628_2823"
            },
            "brzs": {
                "key": "brzs",
                "name": "病人主诉",
                "value": "无明显不适"
            },
            "ct": {
                "key": "ct",
                "name": "查体",
                "value": "神清，精神可。皮肤巩膜无黄染。双侧颈部、锁骨上淋巴结未触及肿大。颈软，胸廓无畸形，两肺呼吸音清，未闻及明显干湿性啰音。心律齐，心脏各瓣膜区未闻及病理性杂音。腹软，无压痛及反跳痛，肝脾肋下未触及。双下肢无水肿。四肢肌力正常，双侧巴氏征阴性。"
            },
            "jcjyjg": {
                "key": "jcjyjg",
                "name": "检查检验结果",
                "value": "暂无更新"
            },
            "zd": {
                "key": "zd",
                "name": "诊断",
                "value": "1.小细胞肺癌（广泛期 cTxNxM1） 2.恶性肿瘤靶向治疗 肺腺癌，cTxN3M1c IVB期，EGFR21外显子L858R点突变 肺部肿瘤消融术后 恶性肿瘤化学治疗后 2.高血压 3.肝囊肿 4.肾囊肿 5.动脉粥样硬化 主动脉硬化 双下肢动脉斑块形成  6.便秘 内痔 7.结肠腺瘤EMR术后 缺血性肠病考虑 8.腰椎间盘突出 颈椎间盘突出 9.甲状腺结节 10.轻度贫血 11.膝关节退行性骨关节病 12.睡眠障碍  13.房性早搏 室性早搏  14.脑梗死 "
            },
            "bqfxyzljh": {
                "key": "bqfxyzljh",
                "name": "病情分析与诊疗计划",
                "value": "患者今日抗肿瘤治疗第3天，无明显不适主诉。继续当前治疗，继观。"
            }
        },
        {
            "bizType": {
                "key": "bizType",
                "name": "bizType",
                "value": "zzyscfjl"
            },
            "doc_time": {
                "key": "doc_time",
                "name": "doc_time",
                "value": "2024-05-17 08:45:44.000"
            },
            "blh": {
                "key": "blh",
                "name": "病历号",
                "value": "98872"
            },
            "jzcs": {
                "key": "jzcs",
                "name": "就诊次数",
                "value": "193"
            },
            "cjsj": {
                "key": "cjsj",
                "name": "创建时间",
                "value": "2024-05-17 08:18:27.000"
            },
            "gxsj": {
                "key": "gxsj",
                "name": "更新时间",
                "value": "2024-05-20 03:08:52.000"
            },
            "blmc": {
                "key": "blmc",
                "name": "病历名称",
                "value": "主治医师查房记录"
            },
            "病历ID": {
                "key": "病历ID",
                "name": "病历ID",
                "value": "98872_34130-0412_20240517161821_4281"
            },
            "brzs": {
                "key": "brzs",
                "name": "病人主诉",
                "value": "无不适"
            },
            "ct": {
                "key": "ct",
                "name": "查体",
                "value": "神清，精神可。皮肤巩膜无黄染。双侧颈部、锁骨上淋巴结未触及肿大。颈软，胸廓无畸形，两肺呼吸音清，未闻及明显干湿性啰音。心律齐，心脏各瓣膜区未闻及病理性杂音。腹软，无压痛及反跳痛，肝脾肋下未触及。双下肢无水肿。四肢肌力正常，双侧巴氏征阴性。"
            },
            "jcjyjg": {
                "key": "jcjyjg",
                "name": "检查检验结果",
                "value": "(2024-05-13)颅脑MRI平扫示：右侧额叶新近脑梗死。脑干、两侧额顶叶皮层下、侧脑室旁散在缺血性改变。脑干及小脑半球小软化灶可能。脑萎缩。附见：右侧乳突少许炎症。部分鼻窦粘膜增厚。C3椎体信号不均匀。"
            },
            "zd": {
                "key": "zd",
                "name": "诊断",
                "value": "1、恶性肿瘤维持性化学治疗 小细胞肺癌（广泛期 cTxNxM1） 2.恶性肿瘤靶向治疗 肺腺癌，cTxN3M1c IVB期，EGFR21外显子L858R点突变 肺部肿瘤消融术后 恶性肿瘤化学治疗后 2.高血压 3.肝囊肿 4.肾囊肿 5.动脉粥样硬化 主动脉硬化 双下肢动脉斑块形成  6.便秘 内痔 7.结肠腺瘤EMR术后 缺血性肠病考虑 8.腰椎间盘突出 颈椎间盘突出 9.甲状腺结节 10.轻度贫血 11.膝关节退行性骨关节病 12.睡眠障碍   13.房性早搏 室性早搏  14.脑梗死  "
            },
            "bqfxyzljh": {
                "key": "bqfxyzljh",
                "name": "病情分析与诊疗计划",
                "value": "入院予苯磺酸氨氯地平片降压，阿托伐他汀钙片降脂稳斑，甲钴胺营养神经，止咳化痰护胃治疗。排除禁忌后于2024-5-10行第三周期依托泊苷 0.12g d1-d3，卡铂 350mg d1方案化疗，化疗顺利，05.15予长效升白预防骨髓抑制。建议地舒单抗/唑来膦酸等治疗，患者拒绝。现患者稍感恶心无呕吐，无头晕头痛不适，请示上级医师后予今日出院。患者头颅磁共振提示新发脑梗死，神经内科会诊：建议监测控制血压血糖，继续普伐他汀钙片口服，加予氯吡格雷片75mg po qd。嘱患者神经内科门诊随诊。 "
            }
        }
    ],
    "ICULHCFJL": [],
    "ZRYSCFJL": [
        {
            "bizType": {
                "key": "bizType",
                "name": "bizType",
                "value": "zryscfjl"
            },
            "doc_time": {
                "key": "doc_time",
                "name": "doc_time",
                "value": "2024-05-17 08:45:44.000"
            },
            "blh": {
                "key": "blh",
                "name": "病历号",
                "value": "98872"
            },
            "jzcs": {
                "key": "jzcs",
                "name": "就诊次数",
                "value": "193"
            },
            "cjsj": {
                "key": "cjsj",
                "name": "创建时间",
                "value": "2024-05-17 08:16:55.000"
            },
            "gxsj": {
                "key": "gxsj",
                "name": "更新时间",
                "value": "2024-05-17 08:45:44.000"
            },
            "blmc": {
                "key": "blmc",
                "name": "病历名称",
                "value": "(副)主任医师查房记录"
            },
            "病历ID": {
                "key": "病历ID",
                "name": "病历ID",
                "value": "98872_32785-141231_20240517161648_9542"
            },
            "brzs": {
                "key": "brzs",
                "name": "病人主诉",
                "value": "无恶心呕吐等不适"
            },
            "ct": {
                "key": "ct",
                "name": "查体",
                "value": "神清，精神可。皮肤巩膜无黄染。双侧颈部、锁骨上淋巴结未触及肿大。颈软，胸廓无畸形，两肺呼吸音清，未闻及明显干湿性啰音。心律齐，心脏各瓣膜区未闻及病理性杂音。腹软，无压痛及反跳痛，肝脾肋下未触及。双下肢无水肿。四肢肌力正常，双侧巴氏征阴性。"
            },
            "jcjyjg": {
                "key": "jcjyjg",
                "name": "检查检验结果",
                "value": "(2024-05-13)血常规：白细胞计数 3.6×10^9/L，中性粒细胞百分比 64.9%，红细胞计数 2.87×10^12/L↓，血红蛋白测定 96g/L↓，血小板计数 205×10^9/L。  "
            },
            "zd": {
                "key": "zd",
                "name": "诊断",
                "value": "1.小细胞肺癌（广泛期 cTxNxM1） 2.恶性肿瘤靶向治疗 肺腺癌，cTxN3M1c IVB期，EGFR21外显子L858R点突变 肺部肿瘤消融术后 恶性肿瘤化学治疗后 2.高血压 3.肝囊肿 4.肾囊肿 5.动脉粥样硬化 主动脉硬化 双下肢动脉斑块形成  6.便秘 内痔 7.结肠腺瘤EMR术后 缺血性肠病考虑 8.腰椎间盘突出 颈椎间盘突出 9.甲状腺结节 10.轻度贫血 11.膝关节退行性骨关节病 12.睡眠障碍  13.房性早搏 室性早搏  "
            },
            "bqfxyzljh": {
                "key": "bqfxyzljh",
                "name": "病情分析与诊疗计划",
                "value": "患者昨日化疗最后一天，患者未诉不适。再次与患者及家属沟通患者骨转移，可行护骨治疗，患者及家属拒绝。继观病情。"
            }
        },
        {
            "bizType": {
                "key": "bizType",
                "name": "bizType",
                "value": "zryscfjl"
            },
            "doc_time": {
                "key": "doc_time",
                "name": "doc_time",
                "value": "2024-05-17 08:45:38.000"
            },
            "blh": {
                "key": "blh",
                "name": "病历号",
                "value": "98872"
            },
            "jzcs": {
                "key": "jzcs",
                "name": "就诊次数",
                "value": "193"
            },
            "cjsj": {
                "key": "cjsj",
                "name": "创建时间",
                "value": "2024-05-17 08:16:11.000"
            },
            "gxsj": {
                "key": "gxsj",
                "name": "更新时间",
                "value": "2024-05-17 08:45:38.000"
            },
            "blmc": {
                "key": "blmc",
                "name": "病历名称",
                "value": "(副)主任医师查房记录"
            },
            "病历ID": {
                "key": "病历ID",
                "name": "病历ID",
                "value": "98872_32785-141231_20240517161606_7595"
            },
            "brzs": {
                "key": "brzs",
                "name": "病人主诉",
                "value": "无明显不适"
            },
            "ct": {
                "key": "ct",
                "name": "查体",
                "value": "神清，精神可。皮肤巩膜无黄染。双侧颈部、锁骨上淋巴结未触及肿大。颈软，胸廓无畸形，两肺呼吸音清，未闻及明显干湿性啰音。心律齐，心脏各瓣膜区未闻及病理性杂音。腹软，无压痛及反跳痛，肝脾肋下未触及。双下肢无水肿。四肢肌力正常，双侧巴氏征阴性。"
            },
            "jcjyjg": {
                "key": "jcjyjg",
                "name": "检查检验结果",
                "value": "(2024-05-09)血常规：白细胞计数 4.4×10^9/L，中性粒细胞百分比 62%，淋巴细胞百分比 24.9%，单核细胞百分比 10.6%↑，血红蛋白测定 100g/L↓；D-二聚体 1.14mg/L↑；凝血功能常规：无殊；生化全套+心肌酶谱：肌酸激酶 258U/L↑，白蛋白 36.6g/L↓，脂肪酶 60.5U/L↑，氯(急) 115mmol/L↑，总钙(急) 1.97mmol/L↓；高敏肌钙蛋白T 0.009ng/ml；B型尿钠肽 20.2pg/ml；尿液分析：无殊；粪便隐血试验阴性；肿瘤标志物（女）：癌胚抗原 26.44ng/ml↑，细胞角蛋白21-1  2.24ng/mL↑；胃泌素释放肽前体 790.34pg/ml↑；"
            },
            "zd": {
                "key": "zd",
                "name": "诊断",
                "value": "1.小细胞肺癌（广泛期 cTxNxM1） 2.恶性肿瘤靶向治疗 肺腺癌，cTxN3M1c IVB期，EGFR21外显子L858R点突变 肺部肿瘤消融术后 恶性肿瘤化学治疗后 2.高血压 3.肝囊肿 4.肾囊肿 5.动脉粥样硬化 主动脉硬化 双下肢动脉斑块形成  6.便秘 内痔 7.结肠腺瘤EMR术后 缺血性肠病考虑 8.腰椎间盘突出 颈椎间盘突出 9.甲状腺结节 10.轻度贫血 11.膝关节退行性骨关节病 12.睡眠障碍  13.房性早搏 室性早搏   "
            },
            "bqfxyzljh": {
                "key": "bqfxyzljh",
                "name": "病情分析与诊疗计划",
                "value": "今完善检查排除禁忌后行第二周期依托泊苷 0.12g d1-d3，卡铂 350mg d1方案化疗，余继续阿美替尼抗肿瘤治疗，苯磺酸氨氯地平片降压，阿托伐他汀钙片降脂稳斑，甲钴胺营养神经，止咳化痰护胃治疗。继观。"
            }
        }
    ],
    "KZRCFJL": [],
    "RCBCJL": []
}'


curl -X POST -d @aa.json http://gbox11.aigauss.com:9601/infer

curl -X --location --request POST 'http://gbox11.aigauss.com:9601/infer' \
--header 'Content-Type: application/json' \
-d @aa.json


"""
