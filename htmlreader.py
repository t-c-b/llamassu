from bs4 import BeautifulSoup
from sys import argv

with open(argv[1]) as fp:
    soup = BeautifulSoup(fp)
    content = soup.main
    for monster in content.find_all("section"):
        name = monster.find(["h2", "h3"]).string
        of = open("data/" + ''.join(c for c in name if c.isalnum()) , "w")

        of.write(f"{name}\n\n")

        table = monster.find("table")
        if table != None:
            if table.thead != None:
                for tr in table.thead.find_all("tr"):
                    headers = tr.find_all("th")

                    of.write("|")
                    for th in headers:
                        of.write(f" {str(th.string or '')} |")
                    of.write("\n|")

                    for i in range(len(headers)):
                        of.write(" --- |")
                    of.write("\n")

            for tr in table.tbody.find_all("tr"):
                of.write("|")
                for td in tr.find_all("td"):
                    of.write(f" {str(td.string or '')} |")
                of.write("\n")

        of.write("\n")

        for p in monster.find_all("p"):
            of.write(''.join(p.strings) + "\n\n")

