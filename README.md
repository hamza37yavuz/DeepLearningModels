# DERİN ÖĞRENME VERİ SETİ ÜZERİNDE KARŞILAŞTIRMALI MODEL İNCELEMESİ

## ÖZET:
Bu çalışmada, AlexNet, VGG, ResNet ve GoogleNet olmak üzere dört farklı derin öğrenme modeli üzerinde inceleme yapılmıştır. Modellerin eğitileceği veri seti, 5 farklı sınıftan oluşmaktadır ve dengesiz bir dağılıma sahiptir. Özellikle sınıf 0 ve 1'deki performansı artırmak amacıyla, bu sınıfların daha iyi temsil edilmesi gerektiği düşünülmüştür. Ancak, model kurulmadan önce bu yönde bir çalışma yapılmamıştır. Model kurulmadan önce veri üzerinde sadece transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) işlemi ile normalleştirme yapılmıştır. 

Çalışmanın amacı, benzer koşullar altında 4 farklı yapay sinir ağının sonuçlarını karşılaştırmaktır. Bu sebeple, belirlenen tüm derin öğrenme ağlarının eğitiminde aynı optimizasyon algoritmaları ve aynı kayıp fonksiyonları kullanılmıştır. Optimizasyon algoritması olarak stokastik gradyan alçalışı (Stochastic Gradient Descent) tercih edilmiştir. Kayıp fonksiyonu olarak CrossEntropyLoss seçilmiştir. Modellerin minimum hata oranına 25 adımda (epoch'ta) ulaşabileceği düşünülmüştür. Aşırı öğrenen sinir ağları için adım sayısı düşürülmüştür. Optimizasyon algoritması olarak mini batch yapısı ile kullanılmıştır. Bu algoritma modelin her seferinde küçük veri parçaları üzerinde güncelleme yapmasını sağlamaktadır. Batch boyutu 64 olarak belirlenmiştir ama eğitimde farklı batch değerleri de kullanılmıştır.  Eğitim sırasında, modellerin son ağırlıkları ve en yüksek doğruluk değerine sahip olan ağırlıkları kaydedilmiştir. Tüm bu işlemler Python yazılım dili ve Pytorch kütüphanesi kullanılarak gerçekleştirilmiştir.

Çalışmanın sonunda, modellerin performansı karmaşıklık matrisleri ve model skorları ile incelenmiş, sonuçlar karşılaştırmalı olarak tablolar halinde sunulmuştur. Çalışmanın kodlarına [buradan](https://www.kaggle.com/code/muhammethamzayavuz/deeplearningmodels) ulaşabilirsiniz.

## ALEXNET:
8 katmanı (5 evrişim katmanı ve 3 tam bağlantılı katman) bulunan bir CNN mimarisidir. Sırasıyla evrişim (Conv2d), aktivasyon, havuzlama ve tam bağlantı katmanlarından oluşmaktadır. Evrişim katmanları, girdi görüntüyü özellik haritalarına dönüştürmektedir. Aktivasyon fonksiyonları (ReLU), non-lineerlik ekler. Havuzlama katmanları, boyut azaltma ve özellik koruma sağlamaktadır. Tam bağlantı katmanları ise sınıflandırma yapmaktadır.

![](https://github.com/hamza37yavuz/DeepLearningModels/blob/main/AlexnetLoss.png)

AlexNet kullanılarak 25 adım boyunca eğitilmiştir. Grafikte eğitim ve doğrulama kayıplarının adımlara göre değişimi görülmektedir. Eğitim kaybı (mavi çizgi) batch yapısı sebebiyle düzenli olarak azalırken, doğrulama kaybı (turuncu çizgi) daha dalgalı ama genel olarak azalma eğilimindedir. Eğitim kaybının doğrulama kaybına göre daha düzenli azalması, modelin eğitim verisini iyi öğrendiğini, ancak doğrulama verisi üzerindeki performansın daha değişken olduğunu gösterir. Genel olarak, modelin hem eğitim hem de doğrulama verisi üzerinde ilerleme kaydettiği gözlemlenmektedir.

Ayrıca başlangıçta görüntüler, transforms.RandomHorizontalFlip() ile %50 olasılıkla yatay olarak çevrilmiş ve transforms.RandomRotation(10) ile rastgele 10 dereceye kadar döndürülmüştür. Bunlara ek olara transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2) kullanılarak görüntünün parlaklığı, kontrastı, doygunluğu ve tonu rastgele değiştirilmiştir. Bu dönüşümler, modelin farklı yönelimlerde ve açılarda görüntülerle eğitilmesini sağlamak amacıyla yapılmıştır. Ama alınan skorlar derece değerlerinin değiştirilmesine rağmen veri çoğaltma olmadan yapılan eğitimden daha düşük skorlar alınmıştır. 

![](https://github.com/hamza37yavuz/DeepLearningModels/blob/main/AlexnetConfusionm.png)

Resim 2’de, modelin sınıflandırma performansı gösterilmektedir. Genel olarak, modelin sınıf 3'teki başarısı yüksektir, ancak diğer sınıfları öğrenme açısından modelin başarısız olduğu söylenebilir. Model daha yüksek (50 adım) ile eğitildiğinde verinin dengesizliği sebebiyle sınıf 3 de doğru sınıflandırma artmakta ama diğer sınıflar için doğru sınıflandırma oranı azalmaktadır.

## VGGNET: 
16 veya 19 katmandan oluşan derin bir CNN mimarisidir. Bizim kullandığımız 16 katmanlı halidir. Bu katmanlar, evrişim katmanları ve tam bağlantılı katmanlar (Linear) olarak iki ana bölüme ayrılmaktadır. Evrişim katmanları, giriş görüntülerinin özelliklerini çıkarmak için kullanılırken, tam bağlantılı katmanlar sınıflandırma yapmak için kullanılmaktadır. Katmanları sayarken, her bir evrişim katmanı ve her bir tam bağlantılı katman bir katman olarak sayılır.

![](https://github.com/hamza37yavuz/DeepLearningModels/blob/main/VGGLoss.png)

VGG öncelikle 25 adım boyunca eğitilmiştir. 25 Adım boyunca grafiklerde hep sabit kayıp değerleri görülmesi sebebiyle sonrasında 5 adım boyunca eğitilerek skor karşılaştırılması yapılmıştır.  Grafik incelendiğinde, eğitim kaybının ilk birkaç adım boyunca hızlı bir şekilde azaldığı ve sonrasında nispeten sabit bir seviyeye ulaştığı görülmektedir. Doğrulama kaybı ise başlangıçta düşüş gösterip daha sonra neredeyse sabit bir seviyede devam etmektedir. Bu durum, modelin eğitim verisi üzerinde iyi bir şekilde öğrenme sağladığını ve doğrulama verisi üzerinde de genel olarak tutarlı performans sergilediğini göstermektedir. 
Modelde aşırı uyum belirtisi gözlenmemektedir. Ama model skorlarının beklenenden düşük olduğu görülmüştür. Bu durumun sebebinin 3 farklı durum olabileceği düşünülmüştür.

1-	Dengesiz Veri

2-	Model Mimarisinin Uygun Olmaması

3-	Hiperparametre Ayarları

Dengesiz veriyle ilgili bir çalışma bu projede yapılmamıştır. Model mimarisinin uygun olmaması projenin konusu değildir. Hiperparametre ayarlarının değiştirilmesi bu model için uygulanmamıştır. Bu model için erken durdurma (early stopping) belirli parametrelerle uygulandığında model 5. adıma geldiğinde sonlanmaktadır. Son aşamada erken durdurma kullanılmamıştır. Ama Resim 4’te 6 adım ile eğitilmiş model gösterilemktedir.

![](https://github.com/hamza37yavuz/DeepLearningModels/blob/main/VGGLoss2.png)

## RESNET18: 
Burada kullanılan ResNet18 modeli toplamda 18 katmandan oluşmaktadır. Katmanlar, evrişim (Conv2d), normalleştirme (BatchNorm2d), ReLU aktivasyon fonksiyonları ve toplama işlemleri içermektedir. Her bir katman, BasicBlock adı verilen bir yapıyı kullanır. Bu yapı, içinde iki evrişim katmanı bulunan ve toplama işlemiyle birleştirilen bir "temel bloktan" oluşmaktadır. Ardından bu temel bloklar, farklı özellik haritası boyutlarına sahip dört katmana yayılmıştır. Katmanların sayısı, her bir katmanın içindeki blok sayısı ve katmanların kendisi dahil edilerek hesaplanmaktadır.

![](https://github.com/hamza37yavuz/DeepLearningModels/blob/main/ResnetLoss.png)

ResNet18 sinir ağı öncelikli olarak 25 adım boyunca eğitilmiştir. Ama 25 adım eğitildiğinde model aşırı öğrenme göstermiştir. Bu sebeple 0.5 dropout kullanılarak 15 adım tekrar eğitilmiştir. Model 15 adım eğitildiğinde aşırı öğrenme göstermemiştir. Ama test kayıplarının orantısız değişimi görülmüştür. Bu grafikte dengeli bir eğitim gerçekleştiği görülmektedir.
15 adımdan fazla eğitildiğinde model eğitim kümesini ezberlemeye gitmektedir. Bu durumu engellemek için veri çoğaltma işlemi yapılabilir. Bu model için veri çoğaltma işlemine başvurulmamıştır.

![](https://github.com/hamza37yavuz/DeepLearningModels/blob/main/ResnetConfusionm.png)

Karmaşıklık matrisine bakıldığında diğer modellere kıyasla daha iyi bir öğrenme sağladığı gözlemlenmektedir. Model sadece 2 numaralı sınıfı doğru bir şekilde öğrenememiştir. Sınıf 0, 1 ve 3 için oldukça iyi tahminler yapmıştır. Sınıf 5 için ise ortalama bir skor elde edildiği söylenebilir.

## GOOGLENET:
22 katmanlık karmaşık bir mimariye sahiptir. Paralel evrişim katmanları ve boyut azaltma modülleri, farklı ölçeklerde özelliklerin keşfedilmesine olanak tanımaktadır. "Inception module" adı verilen yapı, çeşitli filtre boyutlarını ve çekirdek tiplerini bir araya getirmektedir. Bu model transfer öğrenme olarak kullanılmıştır.

![](https://github.com/hamza37yavuz/DeepLearningModels/blob/main/GoogleNetLoss.png)

Googlenet önceden eğitilmiş ağırlıklarla başlayan (pretrained) modeli öncelikle 25 adım boyunca eğitilmiştir. Ama bu eğitimde model aşırı öğrenmiştir. Bu sebeple 0.5 dropout eklenmiş ve adım sayısı 15’e düşürülmüştür. Ama model yine aşırı öğrenme göstermiştir. Bu sebeple veri çoğaltma işlemine başvurulmuştur. transforms.RandomHorizontalFlip() ile %50 olasılıkla yatay olarak çevrilmiş ve transforms.RandomRotation(10) ile rastgele 10 dereceye kadar döndürülmüştür. Bunlara ek olara transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2) kullanılarak görüntünün parlaklığı, kontrastı, doygunluğu ve tonu rastgele değiştirilmiştir. Yapılan işlemler sonrası batch boyutu 32 olacak şekilde güncellenmiştir. Ayrıca model optimizasyonu olarak kullanılan SGD algoritmasında weight_decay parametresi 1e-4 olacak şekilde düzenlenmiştir. Model 15 adım tekrar eğitildiğinde yukardaki kayıp grafiği ortaya çıkmıştır. Yukardaki grafiğe bakıldığında modelin dengeli bir şekilde eğitimi tamamladığı görülmektedir.

Model 15 adımdan fazla eğitildiğinde aşırı öğrenmektedir. Aşırı öğrenmeyi engellemek için yukarda yapılan işlemlerin dışında erken durdurma (early stopping) işlemi uygulanabilirdi. Ama bu model için erken durdurma işlemi kullanılmamıştır.

![](https://github.com/hamza37yavuz/DeepLearningModels/blob/main/GoogleNetConfusionM.png)

Diğer modellerle kıyaslandığında modelimizin (GoogleNet) bütün sınıfları iyi şekilde öğrendiği göze çarpmaktadır. Verinin dengesiz dağılımı sebebiyle sınıf 3 için etkili tahminler yapabilmektedir. Modelin 4 numaralı sınıfı iyi bir şekilde öğrenemediği Resim 8’e bakıldığında görünmektedir. Model skorunu arttırmak için veri setinin dengeli bir şekilde kullanılması denenebileceği düşünülmüştür. Örneğin her sınıftan belirli bir sayıda veri alarak veri çoğaltma işlemi yapıldıktan sonra veri setinin tamamını kullanmadan eğitim gerçekleştirilebilir. Bu durumda model her sınıfı iyi şekilde öğrenecektir. Bu yaklaşım bu model için uygulanmamıştır.

## MODELLERİN SKORLARININ İNCELENMESİ:
| Model      | Accuracy (TEST) | Precision (TEST) | Recall (TEST) | F1_score (TEST) | Accuracy (EĞİTİM) | Precision (EĞİTİM) | Recall (EĞİTİM) | F1_score (EĞİTİM) | Epoch | Augmentation | Model Tipi | DropOut |
|------------|------------------|------------------|---------------|-----------------|-------------------|--------------------|------------------|-------------------|-------|--------------|------------|---------|
| AlexNet    | 0.6622           | 0.6303           | 0.6622        | 0.6322          | 0.6798            | 0.6383             | 0.6798           | 0.6458            | 25    | YOK          | Last       | YOK     |
| AlexNet    | 0.6669           | 0.5567           | 0.6669        | 0.5954          | 0.6693            | 0.5696             | 0.6693           | 0.5973            | 50    | VAR          | Last       | YOK     |
| VGGNet     | 0.6115           | 0.3739           | 0.6115        | 0.4641          | 0.6155            | 0.3789             | 0.6155           | 0.4691            | 25    | YOK          | Best       | YOK     |
| VGGNet     | 0.6115           | 0.3739           | 0.6115        | 0.4641          | 0.6155            | 0.3789             | 0.6155           | 0.4691            | 5     | YOK          | Last       | VAR     |
| ResNet     | 0.6129           | 0.7226           | 0.6129        | 0.6434          | 0.6390            | 0.7399             | 0.6390           | 0.6675            | 15    | YOK          | Last       | VAR     |
| GoogleNet  | 0.8278           | 0.8374           | 0.8278        | 0.8298          | 0.9111            | 0.9123             | 0.9111           | 0.9110            | 15    | VAR          | Last       | VAR     |


Resim 9’daki tabloda, farklı derin öğrenme modellerinin test ve eğitim setlerindeki performansları karşılaştırılmaktadır. Modeller AlexNet, VGGNet, ResNet ve GoogleNet'tir. GoogleNet, veri artırma ile en yüksek performansı göstermektedir. AlexNet, 25 adım ve veri arttırma olmadan, orta seviyede performans sergilemektedir. VGGNet ise hem 25 hem de 5 adım durumlarında düşük performans göstermektedir. ResNet, 15 adım ile orta seviyede sonuçlar vermektedir. Genel olarak, veri artırma kullanılan ve daha yüksek adım sayısına sahip modellerin performansının daha iyi olduğu gözlemlenmiştir.

Bu bulgular ışığında, derin öğrenme projelerinde model seçiminde dikkatli olunması ve veri çoğaltma tekniklerinin uygulanmasının performans üzerinde olumlu etkiler yaratabileceği sonucuna varılmıştır. Ayrıca parametre değişimlerinin modelin çalışma süresini ve skorlarını etkilediği de gözlemlenmiştir. Gelecek çalışmalar, bu modellerin farklı veri setleri ve problemler üzerindeki performansını inceleyerek daha geniş bir perspektif sağlayabilir.
