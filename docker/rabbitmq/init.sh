# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#!/bin/sh

PASSWD_LEN=90
RABBITMQ_PASSWORD=$(< /dev/urandom tr -dc _A-Z-a-z-0-9 | head -c${1:-$PASSWD_LEN};echo;)

# Create Rabbitmq user
(
	rabbitmqctl wait --timeout 120 $RABBITMQ_PID_FILE
	rabbitmqctl add_user $RABBITMQ_USER $RABBITMQ_PASSWORD 2>/dev/null
	rabbitmqctl set_user_tags $RABBITMQ_USER administrator
	rabbitmqctl set_permissions -p / $RABBITMQ_USER  ".*" ".*" ".*"
	rabbitmqctl delete_user 'guest'
	echo "*** User '$RABBITMQ_USER' with password '[hidden]' completed. ***"
) &

echo -e """{
    \"rabbitmq-erlang-cookie\": \"\",
    \"rabbitmq-name\": \"${RABBITMQ_NODENAME}\",
    \"rabbitmq-password\": \"${RABBITMQ_PASSWORD}\",
    \"rabbitmq-user\": \"${RABBITMQ_USER}\",
    \"rabbitmq-vhost\": \"/\",
    \"rabbitmq-port\": 5671,
    \"rabbitmq-host\": \"localhost\",
    \"rabbitmq-cert\": \"ca_certificate.pem\",
    \"rabbitmq-queue-data\": \"NAME-AMS-QUEUE\"
}" > $RABBITMQ_CRED

chmod 600 $RABBITMQ_CRED

# # $@ is used to pass arguments to the rabbitmq-server command.
# # For example if you use it like this: podman run -d <container-name> arg1 arg2,
# # it will be as you run in the container rabbitmq-server arg1 arg2
rabbitmq-server $@
