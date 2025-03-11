# import json

# if __name__ == '__main__':
#     json_root=r'playground/data/llava_v1_5_mix665k.json'
#     with open(json_root, 'r') as f:
#         json_info = json.load(f)
#     save_info=json_info[:1000]+json_info[480000:481000]+json_info[620000:621000]
#     with open('info.json', 'w') as fp:
#         json.dump(save_info, fp, indent=4)

import json

dataset_content = """
[
    {
        "id": "yuanshen-628d-4724-b370-b84de974a19f",
        "image": "yuanshen/1.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWho is in the picture?"
            },
            {
                "from": "gpt",
                "value": "The person in the picture is Nathida, who is a character in the Original God and its derivative works produced by Mihoyo. Her real name is Buyel, the grass god in the \"Earthly Seven rulers\", and is given the nickname of \"Little Lucky Grass King\" by the XuMi people, the youngest of the seven gods today. "
            }
        ]
    }
]
"""

with open("./playground/data/yuanshen.json", "w") as f:
    f.write(dataset_content)