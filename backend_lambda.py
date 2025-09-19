from mangum import Mangum
from backend import app

handler = Mangum(app)
