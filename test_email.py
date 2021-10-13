import smtplib
import pandas as pd
import os
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from datetime import datetime


def send_mail(send_from, send_to, subject, text, files=None,
              server="127.0.0.1"):
    if not isinstance(send_to, list):
        send_to = [send_to]

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=os.path.basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(f)
        msg.attach(part)

    smtp = smtplib.SMTP(server)
    try:
        smtp.sendmail(send_from, send_to, msg.as_string())
        print('send success, ', datetime.now())
    except:
        print('send fail, ', datetime.now())

    smtp.close()


if __name__ == "__main__":
    # fn = ""
    # pd.read_csv(fn, index_col=0)
    tmp_fn = "tmp.csv"
    # pd.to_csv(tmp_fn, encoding='gbk')
    send_from = 'hzhang@quantaeye.com'
    send_to = "zhangheng0101@126.com"
    fmt = "%Y-%m-%d_%H-%M-%S"
    subject = "devices that gps change_" + datetime.now().strftime(fmt)
    text = "devices that gps change_" + datetime.now().strftime(fmt)

    send_mail(send_from, send_to, subject,
              text, files=[tmp_fn], server="127.0.0.1")

