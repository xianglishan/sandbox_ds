#!/bin/bash

json_text='{"api-token":"'${SIGNATE_API_KEY}'"}'
# echo ${json_text}
if [ ! -d ${HOME}/.signate ]; then
    mkdir ${HOME}/.signate
fi
echo ${json_text} > ${HOME}/.signate/signate.json