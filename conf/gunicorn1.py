bind = '0.0.0.0:5001'
timeout = 3000
workers = 2
#worker_class = 'gevent'
accesslog = './logs/access.log'
errorlog = './logs/error.log'
loglevel = 'debug'
preload_app = False