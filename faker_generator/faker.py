import requests
import csv

with open('fake_data_indonesia2.csv', mode='w') as csv_file:
    fieldnames = ['name', 'address', 'birth_data', 'plasticcard']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(123):
        response = requests.get(
            "https://api.namefake.com/indonesian-indonesia/random")
        faked = response.json()
        writer.writerow({
            'name': faked['name'],
            'address': faked['address'],
            'birth_data': faked['birth_data'],
            'plasticcard': faked['plasticcard']
        })
