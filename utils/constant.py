ktp = dict()
ktp['NIK'] = '^NI(K|X)\s*\d*'
ktp['Nama'] = '^N(a|e)m(a|e)(\s*\w*)*$'
ktp['TTL'] = '^(Tempat)?(\/)?Tg(i|l)?(\s)?(Lahir)?(\s*\w*)*$'
ktp['Alamat'] = '^A(l)?amat(\s*\w*?)*$'
ktp['Status'] = '^Status(\s*[A-Za-z]*)*$'
ktp['Pekerjaan'] = '^Pekerjaan(\s*[A-Za-z]*)*$'
ktp['Agama'] = '^(Agama)(\s*[A-Za-z]*){1}'
ktp['Kewarganegaraan'] = '^Kewarganegaraan(\s*[A-Za-z]*)*$'
ktp['Berlaku Hingga'] = '^Berlaku Hingga(\s*[A-Za-z]*)*$'

