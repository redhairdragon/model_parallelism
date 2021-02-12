#!/usr/bin/python3
import boto3
client = boto3.client('ec2')
response = client.describe_instances(
    Filters=[{
        "Name": "tag:usr",
        "Values": ['shen']
    }]
)
private_ips = []
for insts in response['Reservations']:
    for inst in insts["Instances"]:
        if "PrivateIpAddress" in inst:
            private_ips.append(inst["PrivateIpAddress"])
with open("hostfile", "w") as f:
    for ip in private_ips:
        f.write("ubuntu@"+ip+" slot=1\n")
