#!/usr/bin/python3
import boto3
import os


USR_NAME = "ubuntu"

client = boto3.client('ec2')
response = client.describe_instances(
    Filters=[{
        "Name": "tag:usr",
        "Values": ['shen']
    }]
)

for insts in response['Reservations']:
    for inst in insts["Instances"]:
        if "PrivateIpAddress" in inst:
            private_ip = inst["PrivateIpAddress"]
            os.system("scp ~/wikidata/*.npy "+USR_NAME +
                      "@"+private_ip+":~/wikidata")
