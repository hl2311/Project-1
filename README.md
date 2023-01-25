# Project-1
## Chú ý
Nếu muốn biết thêm chi tiết về các chức năng của thuật toán có thể sử dụng bản report(final_1.0).pdf
## Dataset
  Các dataset bao gồm:   
  
                         + AlgerianForestFires.csv

                         + BreastCancerWisconsin.csv

                         + CervicalCancerBehaviorRisk.csv

                         + Humidity.csv
                         
                         + OccupancyEstimation.csv
## Running file (đuôi py)
### EDA.py: 
+ Vai trò: Dùng để in ra những thống kê của từng cột ví dụ như giá trị trung bình, độ lệch chuẩn, số điểm trong cột và số giá trị độc lập trong cột. 
+ Thực hiện: Để chạy file này chỉ cần chạy method PreProcessing() và Tutorial() (đã demo). Sau khi chọn xong sẽ ra các cột trong dataset để chọn, chỉ cần nhập số thứ tự mong muốn là những thông tin trên sẽ được in ra, một vài cột sẽ được in ra đồ thị dạng cột phân bố của các điểm trong cột. Đối với cột nhãn, nếu chọn số thứ tự là cột nhãn sẽ có 2 biểu đồ thể hiện cho cột này đó là biểu đổ tròn và biểu đồ cột nằm ngang.
### CS_IFS.py: 
#### Vai trò: 
Đây là file thuật toán chính đã được tích hợp. Để có thể chạy được file này các bạn chỉ cần chạy 3 method sau:
#### Thực thi:
##### Step 1: 
Đầu tiên khởi tạo model với @param là tên file dataset: model = CS_IFS(filename) (filename này chỉ là filename gốc, không cần điền thêm Train_ và Test_ trước tên file)
##### Step 2: 
Tiếp theo chạy method fit cho tập training set: model.fit(criterion, measure, evaluation, _p) với một vài @param bao gồm 
+ criterion: nếu độ lệch giữa tất cả các trọng số ở hai lần cập nhật liên tiếp mà nhỏ hơn criterion thì sẽ dừng cập nhật trọng số 
+ measure: đây là độ đo dùng để tính khoảng cách giữa các điểm trong dataset bao gồm một vài độ đo sau: "Default" (mặc định), "Manhattan", "Mincowski",   "Ngân", "Hamming", "Hamming3Function"
+ evaluation: đây là phương thức dùng để đánh giá kết quả, sẽ có những cách đánh giá sau: "accuracy"(mặc định), "precificity", "sensitivity", "f1_score", "precision"
+ _p: (tham số này dùng cho thang đo Mincowskin, nếu không dùng thang đo này thì không cần nhập _p) biểu diễn số mũ trong thang đo Mincowski
##### Step 3:
Cuối cùng chỉ cần cần chạy method: model.predict()
##### Output:
Đầu ra sẽ in ra kết quả training và testing của dataset, ngoài ra cũng trả về 2 kết quả này dưới dạng số thập phân nếu muốn dùng kết quả này để hiện thị trong chương trình khác
### TrainAndTestSplitting.py:
#### Vai trò:
Dùng để chia dataset ban đầu thành 2 file training và testing với kích thước theo mong muốn của người dùng
#### Thực thi:
##### Step 1: 
Để khởi tạo model, chỉ cần nhập tên dataset muốn chia thành 2 tập training và testing mong muốn. t = TrainAndTestSplitting(filename)
##### Step 2:
Để thực hiện chia thành 2 tập mong muốn, chúng ta chỉ cần chạy method: t.trainAndTestSplitting(train_size, test_size, method) với những @param như sau:
+ train_size: kích thước tập train trong toàn bộ dataset (0 < train_size < 1)
+ test_size: kích thước tập test trong toàn bộ dataset (0 < test_size < 1 và train_size + test_size = 1)
+ method: có 2 phương pháp chia dữ liệu chính đó là "random" (mặc định) và "stratified". Với "random" chúng ta sẽ chia tập dữ liệu ngẫu nhiên mà không quan tâm đến sự phân bố của nhãn trong tập training và testing. Với "stratified" chúng ta sẽ chia tập training và testing đảm bảo theo độ phân bố trong tập dataset ban đầu (Ví dụ: Nếu ban đầu có 2 lớp A chiếm 30% và B chiếm 70% trong tập dataset ban đầu thì khi chia tập training và testing thì phần trăm của từng lớp trong mỗi tập đều là 30% và 70%)
# The end
