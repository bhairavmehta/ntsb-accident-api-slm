from ntsb_api.docket import parse_docket_html
from ntsb_api.normalize import normalize_case


def test_search_api_shape():
    rec = {"Fields": [{"FieldName": "NtsbNo", "Values": ["DCA20XX001"]},
                      {"FieldName": "EventDate", "Values": ["2020-02-03T00:00:00"]},
                      {"FieldName": "Mkey", "Values": ["101234"]},
                      {"FieldName": "Make", "Values": ["BOEING"]},
                      {"FieldName": "HighestInjury", "Values": ["Fatal"]}]}
    c = normalize_case(rec)
    assert c["ntsb_number"] == "DCA20XX001" and c["event_date"] == "2020-02-03"
    assert c["mkey"] == 101234 and c["make"] == "BOEING" and c["highest_injury"] == "Fatal"
    assert c["links"]["report_pdf"].endswith("/101234/pdf")


def test_export_shape():
    rec = {"cm_ntsbNum": "ERA21LA001", "cm_mkey": 5, "cm_eventDate": "2021-01-01T10:00:00Z",
           "cm_probableCause": "Loss of control.", "cm_findings": [{"findingText": "Aircraft-Control", "causeFactor": "C"}],
           "cm_vehicles": [{"make": "PIPER", "model": "PA-28", "regulationFlightConductedUnder": "Part 91"}]}
    c = normalize_case(rec)
    assert c["make"] == "PIPER" and c["far_part"] == "Part 91"
    assert c["findings"][0]["cause_factor"] == "C" and c["probable_cause"] == "Loss of control."


def test_docket_parser():
    html = ('<table><tr><td>1</td><td><a href="Document/docBLOB?ID=99&FileExtension=pdf&FileName=x">'
            'Operations Factual</a></td><td>12</td></tr></table>')
    items = parse_docket_html(html, "https://data.ntsb.gov/Docket/")
    assert items[0]["url"].startswith("https://data.ntsb.gov/Docket/Document/docBLOB?ID=99")
    assert items[0]["file_type"] == "PDF" and items[0]["pages"] == 12 and items[0]["title"] == "Operations Factual"
