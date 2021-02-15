#!/bin/bash
host_filename='hostfile'
result_filename='merged_logs'
log_filename='pipeline_profile_log'
rm -f $result_filename
touch $result_filename
while read line; do 
    host=`echo $line | awk -F\  '{print $1}'`
    ssh -n $host "cat `pwd`/$log_filename" >>$result_filename
    echo $host
done < $host_filename