#!/bin/sh
# Substitute only INTERNAL_API_KEY (not nginx variables like $host, $scheme, etc.)
envsubst '${INTERNAL_API_KEY}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf
exec "$@"
