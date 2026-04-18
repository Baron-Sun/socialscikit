import http.server
import os
os.chdir(os.path.dirname(__file__))
http.server.HTTPServer(("", 8888), http.server.SimpleHTTPRequestHandler).serve_forever()
