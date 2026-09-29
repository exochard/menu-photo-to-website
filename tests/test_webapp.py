import base64
import json

import webapp


def call(method, path, body=None):
    event = {"requestContext": {"http": {"method": method}}, "rawPath": path,
             "body": None if body is None else json.dumps(body), "isBase64Encoded": False}
    out = webapp.handler(event, None)
    return out["statusCode"], out


def test_page_and_demo_assets_are_served():
    status, out = call("GET", "/")
    assert status == 200 and "Menu photo to website" in out["body"]
    status, out = call("GET", "/demo.jpg")
    assert status == 200 and out["isBase64Encoded"] and base64.b64decode(out["body"])[:2] == b"\xff\xd8"


def test_demo_plan_then_apply_rebuilds_the_page_with_escaped_text():
    status, out = call("POST", "/plan", {})
    plan = json.loads(out["body"])
    assert status == 200 and plan["status"] == "proposals"
    assert {c["name"] for c in plan["proposals"]} == {"Cannolo siciliano", "Braciole di maiale"}
    evil = dict(plan["proposals"][0], kind="new_item", group="Dolci", name="<script>x</script>")
    status, out = call("POST", "/apply", {"approved": plan["proposals"] + [evil]})
    result = json.loads(out["body"])
    assert status == 200 and "12,50 €" in result["page"]
    assert "<script>x</script>" not in result["page"] and "&lt;script&gt;" in result["page"]


def test_bad_input_is_a_400_not_a_crash():
    assert call("POST", "/plan", {"photo": "not base64!!"})[0] == 400
    assert call("POST", "/plan", {"photo": base64.b64encode(b"hello").decode()})[0] == 400
    assert call("POST", "/plan", {"config": "pages: 3"})[0] == 400
    assert call("POST", "/apply", {"approved": [{"kind": "price"}]})[0] == 400
    assert call("POST", "/nope", {})[0] == 404
