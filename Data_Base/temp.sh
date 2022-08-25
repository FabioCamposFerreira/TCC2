#!/bin/bash

origin_folder="DATA 3/"
destin_folder="Data_Base_Cedulas/"
origin_folder_len=${#origin_folder}
next_2="31"
next_5="33"
next_10="23"
next_20="27"
next_50="28"
next_100="13"

for file in "${origin_folder}"*.jpg; do
    new_name=${file:$origin_folder_len:100}
    new_name=${new_name//.*/}
    if test $new_name -eq 2; then
        new_name=${destin_folder}${new_name}.${next_2}.jpg
        next_2=$(expr $next_2 + 1)
    else
        if test $new_name -eq 5; then
            new_name=${destin_folder}${new_name}.${next_5}.jpg
            next_5=$(expr $next_5 + 1)
        else
            if test $new_name -eq 10; then
                new_name=${destin_folder}${new_name}.${next_10}.jpg
                next_10=$(expr $next_10 + 1)
            else
                if test $new_name -eq 20; then
                    new_name=${destin_folder}${new_name}.${next_20}.jpg
                    next_20=$(expr $next_20 + 1)
                else
                    if test $new_name -eq 50; then
                        new_name=${destin_folder}${new_name}.${next_50}.jpg
                        next_50=$(expr $next_50 + 1)
                    else
                        if test $new_name -eq 100; then
                            new_name=${destin_folder}${new_name}.${next_100}.jpg
                            next_100=$(expr $next_100 + 1)
                        fi
                    fi
                fi
            fi
        fi
    fi
    cp -i "$file" "$new_name"
    echo "copiando $file para $new_name"
done