# Used-Cars-Prediction-Project
This project is used to help stakeholders predict used car prices
### **Contents**
1. Business Problem Understanding
2. Data Understanding
3. Exploratory Data Analysis
4. Data Cleaning, Feature Selection and Feature Engineering
5. Exploratory Data Analysis after Data Cleaning
6. Modelling
7. Conclusion and Recommendation

## **1. Business Problem Understanding**

### **1.1 Context**

Ada banyak kategori atau jenis kendaraan yang beredar di pasar, seperti *city car, coupe, sedan, support car, hatchback, station wagon, convertible, pickup truck, and sport-utility vehicle (SUV)*. Setiap jenis kendaraan ini memiliki karakteristik, kegunaan, dan harga yang bervariasi, tergantung pada jenis kendaraan tersebut [[1]](https://doi.org/10.35940/IJEAT.A1042.1291S319). Pembelian kendaraan bekas merupakan opsi yang menarik bagi masyarakat di banyak negara karena harganya lebih terjangkau dan memberikan peluang untuk menjual kembali kendaraan tersebut setelah beberapa waktu, yang bisa menghasilkan keuntungan [[2]](https://doi.org/10.29027/IJIRASE.v4.i3.2020.686-689)[[3]](https://doi.org/10.1109/ICBIR.2018.8391177).

Di Arab Saudi, permintaan akan kendaraan bekas meningkat karena beberapa faktor, salah satunya adalah peningkatan pajak dari 5% menjadi 15%, yang dimulai pada bulan Juli 2020 hingga saat ini [[4]](https://doi.org/10.32602/jafas.2021.017). Keputusan ini telah memengaruhi preferensi pembeli, membuat kendaraan bekas menjadi pilihan yang lebih menarik dibandingkan dengan pembelian kendaraan baru. Menjual kendaraan bekas bisa menjadi tantangan karena menentukan nilai wajarnya menjadi lebih sulit, terutama dengan harga yang sangat bergantung pada karakteristik khusus yang hanya pemilik kendaraan yang tahu. Untuk memprediksi harga kendaraan dengan akurat, diperlukan keahlian ahli data [[5]](https://shubh17121996.medium.com/used-car-price-prediction-using-supervised-machine-learning-ea9dace76686).

Tentu saja, faktor yang paling signifikan dalam menentukan harga sebuah kendaraan adalah model kendaraan, usia, jumlah kilometer yang telah ditempuh, merek, tenaga mesin, jenis bahan bakar yang digunakan, dan efisiensi bahan bakar per mil. Terutama, jenis bahan bakar seringkali menjadi faktor yang memengaruhi harga, mengingat fluktuasi harga bahan bakar yang sering terjadi [[6]](https://doi.org/10.18421/TEM81-16). Selain itu, perbedaan dalam fitur seperti jenis transmisi, warna eksterior, fitur keselamatan, jumlah pintu, dimensi, kondisi udara, interior, dan ketersediaan navigasi dapat memengaruhi harga kendaraan [[6]](https://doi.org/10.18421/TEM81-16). Karena banyak faktor dan fitur yang memengaruhi harga kendaraan dan memerlukan banyak waktu dan penilaian manusia, teknik *machine learning* dapat digunakan untuk mengembangkan model yang dapat memprediksi harga kendaraan bekas. Oleh karena itu, banyak penelitian sebelumnya telah mengkaji tugas memprediksi harga kendaraan bekas dengan memanfaatkan *machine learning* dan beberapa di antaranya telah mencapai tingkat akurasi hingga 93% [[7]](https://www.irjmets.com/uploadedfiles/paper/volume3/issue_3_march_2021/6681/1628083284.pdf).

Hingga saat ini, penelitian belum menggunakan data dunia nyata dari Arab Saudi untuk memprediksi harga kendaraan bekas di Arab Saudi. Oleh karena itu, tujuan penelitian ini adalah memberikan kontribusi pada industri otomotif Arab Saudi dengan memanfaatkan kumpulan data dunia nyata dari Arab Saudi yang disediakan oleh platform Kaggle untuk mengembangkan model *machine learning* yang dapat memprediksi harga kendaraan bekas. Ini memiliki dampak penting dalam ruang pamer kendaraan dan juga bermanfaat bagi masyarakat yang ingin menjual kendaraan mereka dengan harga yang adil. Data tersebut diperoleh dari platform Syarah selama tahun 2021, yang memungkinkan pengguna untuk mengiklankan kendaraan bekas mereka. Kumpulan data ini setelah *data cleaning* terdiri dari 2952 baris dan mencakup 10 kolom yaitu *Make, Type, Year, Origin, Options, Engine Size, Gear Type, Mileage, Region, Price*. Fitur-fitur ini, berdasarkan penelitian dan pengetahuan industri otomotif, merupakan faktor-faktor kunci yang memengaruhi harga kendaraan, seperti *Engine Size, Year, dan Make*. Rangkaian langkah eksperimen terdiri dari data understanding, data cleaning, analisis data, pembuatan model, dan evaluasi model.

### **1.2 Problem Statement**

Permasalahan ini akan memiliki dampak langsung baik pada perusahaan yaitu mereka yang bergerak dibidang jual-beli mobil bekas maupun konsumen yaitu mereka yang ingin membeli atau menjual mobil mereka ke perusahaan. Permasalahan terbesar yang dihadapi adalah bagaimana cara menentukan harga mobil bekas agar tidak menjualnya dengan harga terlalu tinggi atau terlalu rendah. Overprice akan membuat calon pembeli tidak tertarik untuk membeli mobil tersebut, sedangkan underprice akan membuat calon penjual menjadi rugi.

Untuk mengatasi masalah tersebut, yaitu menentukan harga mobil bekas berdasarkan Engine Size, Year, Make, Type dan lain-lain dibutuhkan seorang profesional yang memahami kondisi dan skema harga mobil bekas yang tepat berdasarkan pengalamannya. Namun, gaji seorang professional penilai kendaraan bekas dalam hal ini sering disebut sebagai `Vehicle Inspector` sangatlah mahal. Di kutip dari [salaryexpert.com](https://www.salaryexpert.com/salary/job/vehicle-inspector/saudi-arabia#:~:text=The%20average%20vehicle%20inspector%20salary%20in%20Saudi%20Arabia,or%20an%20equivalent%20hourly%20rate%20of%2064%20%D8%B1.%D8%B3.%E2%80%8F.) rata-rata gaji seorang Vehicle Inspector sekitar 133,937 SAR/tahun. Perlu diingat bahwa angka tersebut untuk 1 orang Vehicle Inspector, banyaknya karyawan yang bekerja sebagai Vehicle Inspector tergantung pada beberapa faktor diantaranya Volume Kendaraan, Lokasi dan Cabang serta Jenis Kendaraan. Semakin banyak volume kendaraan, cabang perusahaan serta jenis mobil yang perlu diperiksa, tentunya memerlukan lebih banyak seorang Vehicle Inspector.

Mahalnya gaji seorang Vehicle Inspector tentunya berdampak langsung baik bagi perusahaan juga bagi konsumen. Perusahaan tentu tidak ingin rugi, sehingga solusi yang mungkin dimiliki perusahaan adalah membeli mobil konsumen dengan harga yang lebih murah dan menjualnya dengan harga yang lumayan mahal sehingga bisa menutupi biaya operasional agar perusahaan tetap mendapatkan keuntungan.

### **1.3 Goals**

Berdasarkan problem statement diatas, maka syarah.com selaku perusahaan yang bergerak dibidang jual-beli mobil bekas diharapkan bisa memanfaatkan data historis untuk kemudian dikembangkan menjadi model machine learning yang mampu memprediksi harga mobil bekas dengan tingkat kesalahan yang kecil. Model yang dikembangkan diharapkan mampu membantu para stakeholder dalam menentukan harga mobil bekas dan mengurangi biaya operasional setidaknya lebih kecil dari gaji seorang Vehicle Inspector yaitu 133,937 SAR / Tahun.

### **1.4 Analytic Approach**

Tahapan awal adalah melakukan analisis komprehensif pada seluruh dataset untuk mengidentifikasi pola dan perbedaan antara berbagai fitur yang tersedia. Langkah selanjutnya adalah mengembangkan sebuah model regresi yang akan menjadi alat bantu bagi para stakeholder dalam menentukan prediksi harga mobil bekas. Pemilihan model regresi akan didasarkan pada matrix evaluation yang paling optimal, sehingga kita dapat menentukan model akhir dari teknik machine learning yang akan digunakan.

### **1.5 Metric Evaluation**

Dalam proses pembersihan data, kami tidak menghapus seluruh data yang memiliki outlier. Oleh karena itu, data tetap akan memiliki beberapa outliers. Karena kondisi ini, model ini akan menggunakan metrik evaluasi yang tidak sensitif terhadap outliers. Metrik seperti Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan Root Mean Squared Percentage Error (RMSPE) tidak cocok digunakan pada dataset ini karena sensitif terhadap data outliers metrik-metrik tersebut juga kurang cocok digunakan pada variabel target yang memiliki ragam dan rentang nilai yang sangat besar seperti mata uang.

Sebagai gantinya, metrik evaluasi yang akan digunakan adalah metrik-metrik yang tahan dan mampu menghandle outlier dengan baik. Adapun matrix evaluation yang kami gunakan untuk mengukur kinerja model adalah sebagai berikut:

1. `Mean Absolute Error (MAE)`: MAE mengukur rata-rata dari selisih absolut antara nilai aktual dan nilai yang diprediksi oleh model. Nilai MAE yang lebih rendah menunjukkan kinerja yang lebih baik. Alasan kami menggunakan MAE adalah karena variabel target dalam hal ini adalah variabel Price memiliki outlier sehingga kami memutuskan untuk menggunakan MAE dan MAE merupakan metrics yang kurang sensitif terhadap pencilan/outlier. MAE tahan terhadap outlier karena mengambil nilai mutlak dari selisih antara nilai aktual dan nilai yang diprediksi, sehingga outlier dengan kesalahan besar hanya memberikan kontribusi yang setara tanpa memberikan "bobot" ekstra pada kesalahan tersebut.
2. `Mean Absolute Percentage Error (MAPE)`: MAPE mengukur rata-rata dari selisih persentase antara nilai aktual dan nilai yang diprediksi oleh model. Nilai MAPE yang lebih rendah menunjukkan kinerja yang lebih baik. Alasan kami menggunakan MAPE sama seperti MAE yaitu karena pada variabel target (Price) memiliki outlier dan MAPE kurang sensitif terhadap outlier. MAPE tahan terhadap outlier karena mengukur kesalahan relatif sebagai persentase, dan oleh karena itu, outlier dengan kesalahan besar tidak memiliki dampak yang sangat besar pada metrik ini. MAPE mempertimbangkan perbandingan relatif antara nilai aktual dan nilai yang diprediksi, sehingga outlier dengan nilai besar tidak mengubah proporsi kesalahan yang signifikan. Ini membuat MAPE lebih tahan terhadap dampak outlier yang signifikan pada hasil evaluasi.
3. `Root Mean Squared Logarithmic Error (RMSLE)`: RMSLE adalah metrik yang mengukur perbedaan antara logaritma alami dari nilai aktual dan logaritma alami dari nilai yang diprediksi oleh model. Alasan kami mempertimbangkan matrix ini adalah karena RMSLE juga cenderung lebih tahan terhadap outlier. Matrix ini tahan terhadap outlier karena perbedaan persentase antara nilai aktual dan nilai prediksi diukur setelah transformasi logaritma, sehingga outlier yang mungkin memiliki dampak besar pada metrik lain menjadi lebih terkendali. Namun karena mengambil logaritma data, hasil RMSLE cenderung lebih sulit untuk diinterpretasikan secara langsung dalam konteks asli data, sehingga RMSLE hanya kami gunakan sebagai pertimbangan dalam memilih model terbaik pada benchmarking model.
4. `R-squared (R2)`: R2 mengukur seberapa besar variasi nilai Y yang dapat dijelaskan oleh model. Nilai yang lebih tinggi menunjukkan kinerja yang lebih baik. Sebenarnya R2 kurang cocok digunakan dalam model ini, karena terdapat data outlier pada variabel target dan variabel target memiliki ragam dan rentang nilai yang sangat besar, sehingga matrix ini hanya kami gunakan sebagai pertimbangan dalam memilih model terbaik pada benchmarking model.

## **7. Conclusion and Recommendation**

### **7.1 Conclusion**

Model XGBoost yang menjadi final model dalam experimen ini diukur menggunakan matrix evaluation MAE dan MAPE. 
Berdasarkan matrix evaluation, model ini memiliki error sekitar 10722 SAR (dari MAE) atau sebesar 19% (dari MAPE) prediksi akan melenceng.

Kami telah melakukan research untuk mencari tahu berapa rata-rata kesalahan manusia dalam memprediksi harga mobil bekas, namun kami tidak berhasil menemukannya karena hal ini sangat bergantung pada banyak faktor, termasuk pengalaman individu, sumber data yang digunakan, metode yang digunakan, dan variabilitas pasar mobil bekas yang berbeda-beda. Karena keterbatasan sumber daya, kami tidak bisa membandingkan lebih akurat mana antara menggunakan model yang kami buat atau manusia dalam memprediksi harga mobil bekas.

Karena kami tidak bisa membandingkan antara model dan manusia dalam memprediksi, maka kami hanya akan melakukan perbandingan besarnya biaya yang dikeluarkan antara menggunakan model machine learning dan menggunakan manusia dalam memprediksi harga mobil bekas.

**Perhitungan menggunakan model machine learning**
- Model Infrastructure = 33768 SAR
- Data Support = 25326 SAR
- Engineering/Deployment = 33768 SAR
- Total pengeluaran dalam 1 tahun = 92862 SAR / tahun
Sumber [www.phdata.io](https://www.phdata.io/blog/what-is-the-cost-to-deploy-and-maintain-a-machine-learning-model/)

**Perhitungan menggunakan manusia**
- Mempekerjakan 3 penilai manusia dengan gaji tahunan masing-masing 133937 SAR. 
- Total pengeluaran dalam 1 tahun = 401811 SAR.
Sumber [www.salaryexpert.com](https://www.salaryexpert.com/salary/job/vehicle-inspector/saudi-arabia#:~:text=The%20average%20vehicle%20inspector%20salary%20in%20Saudi%20Arabia,or%20an%20equivalent%20hourly%20rate%20of%2064%20%D8%B1.%D8%B3.%E2%80%8F.)

Dengan menggunakan model machine learning, setidaknya perusahaan mampu menghemat biaya sebesar 308.949 SAR / tahun

### **7.2 Recommendation**

**1. Rekomendasi Penggunaan Model**

* Model ini hanya bisa digunakan pada range harga 10 ribu hingga 180,5 ribu SAR. Di luar range harga tersebut, hasil prediksi akan bias.
* Model yang kami buat memiliki performa yang baik pada range harga mobil mulai dari 20 ribu SAR hingga 150 ribu SAR karena nilai MAPE yang didapat mendekati nilai MAPE pada seluruh data uji. 
* Model ini memiliki performa yang buruk ketika range prediksi berada pada kisaran harga 10 hingga 20 ribu SAR, di mana nilai MAPE mencapai 57%.
* Untuk range harga mobil pada kisaran 150 hingga 180,5 ribu SAR sebenarnya tidak terlalu buruk karena nilai MAPE yang didapat hanya sebesar 22.47% (still acceptable)

**2. Rekomendasi untuk bisnis**

Fitur `Engine_Size`, `Year` dan `Make` merupakan fitur yang paling penting dalam menentukan harga penjualan mobil bekas, sehingga fitur-fitur ini harus diperhatikan.

**3. Rekomendasi Pengembangan Model untuk Penelitian Selanjutnya**

1. Menambahkan fitur-fitur baru seperti `jenis bahan bakar` (bensin/solar), `warna mobil`, dan `jenis mobil` (city car, coupe, sedan, support car, hatchback, station wagon, convertible, pickup truck, and sport-utility vehicle (SUV)) karena fitur-fitur tersebut seringkali memengaruhi harga mobil, terutama jenis bahan bakar seringkali menjadi faktor yang memengaruhi harga, mengingat fluktuasi harga bahan bakar yang sering terjadi [[6]](https://doi.org/10.18421/TEM81-16)
2. Menambah lebih banyak data. Saat model ini dibuat hanya terdiri dari 2952 baris data. Dalam machine learning, memiliki lebih banyak data umumnya dapat meningkatkan kinerja model. Dengan lebih banyak data, model akan memiliki lebih banyak sampel untuk belajar dan beradaptasi, yang dapat meningkatkan akurasi prediksi. Namun, harus dipastikan bahwa data tambahan yang  diperoleh adalah representatif dan berkualitas baik karena kualitas data lebih penting daripada kuantitas.
