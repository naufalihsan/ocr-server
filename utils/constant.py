ktp = dict()
ktp['NIK'] = r'^NI(K|X)\s*(\w)*$'
ktp['Nama'] = r'^N(a|e)m(a|e)(\s*\w*)*$'
ktp['TTL'] = r'^(T(e|o)mpa(t|l))?(\/|\s)?Tg(i|l)?(i)?(\s)?(La(h|b)?)(ir|w)?(\s*(\w|-)*)*$'
ktp['Alamat'] = r'^A(l)?(a|t)?mat(\s*\w*?)*$'
ktp['Pekerjaan'] = r'^Peker(j)?aan(\s*[A-Za-z]*)*$'
ktp['Agama'] = r'^(Agama)(\s*[A-Za-z]*){1}$'