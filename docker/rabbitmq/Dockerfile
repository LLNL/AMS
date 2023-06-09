FROM rabbitmq:3.10

ENV RABBITMQ_CONFIG_FILE /etc/rabbitmq/rabbitmq.conf
ENV RABBITMQ_PID_FILE /var/lib/rabbitmq/mnesia/rabbitmq
# ENV RABBITMQ_NODENAME ams-broker@localhost
ENV RABBITMQ_USE_LONGNAME false
ENV RABBITMQ_USER ams-user
# ENV RABBITMQ_LOGS=-

ADD init.sh /init.sh
RUN chmod +x /init.sh

RUN echo "listeners.tcp = none\n\
          channel_max = 2047\n\
          max_message_size = 134217728\n\
          loopback_users.guest = true\n\
          listeners.ssl.default = 5671\n\
          ssl_options.versions.1 = tlsv1.2\n\
          ssl_options.cacertfile = /certs/ca_certificate.pem\n\
          ssl_options.certfile = /certs/server_certificate.pem\n\
          ssl_options.keyfile = /certs/server_key.pem\n\
          ssl_options.fail_if_no_peer_cert = false\n\
          ssl_options.verify = verify_none" >> $RABBITMQ_CONFIG_FILE

CMD ["/init.sh"]

