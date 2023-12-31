import gspread

sa = gspread.service_account()
sh = sa.open("Költségvetés teszt")

wks = sh.worksheet("2023 tételes")

# import pdb;pdb.set_trace()
# wks.acell("C3").value
# wks.update("C3", "-444")
# wks.update("C4", -1444)

# wks.update("D3", "utazás")

# wks.add_rows(4)

wks.insert_row([4, 11, -2500, "autó", "tisztítás", "Budapest", "CIB kártya", "-"], 3)
