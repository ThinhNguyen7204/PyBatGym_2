

## Thiết kế kỹ thuật của giải pháp

### 1. Nguyên tắc thiết kế tổng thể

Giải pháp PyBatGym được thiết kế theo hướng tách biệt rõ giữa **mô phỏng scheduling**, **môi trường học tăng cường**, và **quản lý thí nghiệm**. Nguyên tắc cốt lõi là không để logic của simulator xâm lấn trực tiếp vào phần lõi của environment, đồng thời không để logic huấn luyện hoặc benchmark làm rối phần tương tác RL cơ bản. Với cách tiếp cận này, hệ thống được tổ chức như một lớp trung gian chuẩn hóa giữa policy và simulator backend, thay vì là một extension ad-hoc gắn chặt với một engine cụ thể.

Thiết kế này xuất phát từ ba yêu cầu kỹ thuật chính. Thứ nhất, môi trường phải tương thích với cách làm việc của các RL framework phổ biến thông qua các khái niệm `reset`, `step`, observation, reward, done và info. Thứ hai, môi trường phải bám sát bản chất event-driven của scheduling simulation, tức agent chỉ được gọi tại những decision point có ý nghĩa. Thứ ba, hệ thống phải hỗ trợ được workflow nghiên cứu, nghĩa là phải lưu được cấu hình, log được kết quả và cho phép đánh giá lặp lại trên cùng điều kiện. Đây chính là ba trụ cột đã được nhấn mạnh trong mục tiêu và tiêu chí đánh giá của đề tài.

Từ các nguyên tắc đó, kiến trúc kỹ thuật của PyBatGym không được xây theo dạng một lớp environment đơn khối. Nếu toàn bộ logic đều dồn vào một lớp trung tâm, hệ thống sẽ rất khó kiểm thử, khó giải thích và khó bảo trì. Thay vào đó, giải pháp được chia thành các mô-đun chuyên trách, mỗi mô-đun xử lý một loại trách nhiệm riêng: cấu hình, tích hợp simulator, quản lý episode, mã hóa observation, xử lý action, tính reward, ghi log và benchmark. Environment Core đóng vai trò điều phối giữa các mô-đun đó.

---

### 2. Kiến trúc phân lớp của hệ thống

Ở mức kỹ thuật, PyBatGym có thể được nhìn như một hệ thống gồm bốn lớp lớn.

Lớp đầu tiên là **Simulator Backend Layer**. Đây là nơi chứa simulator engine thực tế dùng để mô phỏng hệ thống scheduling. Trong phạm vi hiện tại, lớp này được hiện thực bằng BatSim. Tuy nhiên, BatSim không được coi là lõi logic của đề tài; nó chỉ là một backend thực tế mà hệ thống dùng để kiểm chứng tính khả thi. Ở góc nhìn kiến trúc, lớp này chỉ cần cung cấp được trạng thái mô phỏng, xử lý được các quyết định scheduling và cho phép môi trường tiến đến decision point tiếp theo. Điều đó cho phép PyBatGym giữ được định vị là framework event-driven environment, thay vì bị xem là wrapper dành riêng cho BatSim.

Lớp thứ hai là **Integration Layer**, thường được hiện thực bằng một **Simulator Adapter**. Đây là lớp cách ly mọi chi tiết đặc thù của backend simulator khỏi phần lõi environment. Thay vì để Environment Core gọi trực tiếp các API hoặc quy trình điều khiển riêng của BatSim, adapter sẽ đảm nhận việc reset mô phỏng, nạp workload và platform, áp dụng quyết định scheduling, truy vấn queue, truy vấn tài nguyên và phát hiện decision point. Cách làm này có hai lợi ích lớn: nó giúp phần lõi của PyBatGym gọn hơn, và nó giữ được tư duy backend-agnostic ở mức kiến trúc dù hiện tại mới chỉ tích hợp một backend.

Lớp thứ ba là **Environment Core Layer**. Đây là tầng trung tâm của giải pháp. Nó hiện thực semantics của một RL environment, bao gồm reset, step, quản lý observation, reward, episode termination và phối hợp giữa agent với simulator adapter. Tầng này không trực tiếp hiểu chi tiết simulator, cũng không trực tiếp huấn luyện policy. Nó chỉ đảm nhận vai trò diễn dịch một scheduling simulation thành một chu trình tương tác mà RL framework có thể sử dụng.

Lớp thứ tư là **Experiment Support Layer**. Đây là lớp phục vụ nghiên cứu, gồm cấu hình thí nghiệm, logging, artifact export, benchmark runner và các tiện ích đánh giá. Lớp này cực kỳ quan trọng với đề tài, vì nếu thiếu nó, PyBatGym chỉ là một environment chạy được, chưa phải một research framework tối thiểu. Theo slide deck, configuration files, episode metrics, learning curves, scheduler comparison tables và random seed logging đều là artifact được kỳ vọng phải hỗ trợ.

---

### 3. Các mô-đun kỹ thuật chính

#### 3.1. Experiment Configuration Manager

Mô-đun cấu hình là điểm khởi đầu của toàn bộ hệ thống. Nó lưu và cung cấp mọi tham số cần thiết để tái dựng một lần chạy, bao gồm workload trace, platform description, seed, số lượng episode, định nghĩa observation schema, định nghĩa action space, công thức reward, chế độ logging và đường dẫn xuất artifact. Mô-đun này có vai trò như “nguồn chân lý duy nhất” của một lần thí nghiệm.

Việc tách mô-đun này riêng ra là quyết định rất quan trọng về mặt kỹ thuật. Nếu cấu hình bị hard-code rải rác trong từng script hoặc từng lớp, hệ thống sẽ nhanh chóng mất tính tái lập. Trong khi đó, một framework nghiên cứu tối thiểu bắt buộc phải tái hiện được thí nghiệm từ seed và config. Vì vậy, Experiment Configuration Manager không chỉ là phần tiện ích, mà là nền tảng cho reproducibility.

Ở mức thiết kế, mô-đun này nên hỗ trợ đọc cấu hình từ file có cấu trúc, ví dụ YAML hoặc JSON. Sau khi nạp cấu hình, nó cần kiểm tra tính hợp lệ của các tham số cơ bản, rồi cung cấp một object thống nhất cho các mô-đun khác sử dụng. Cấu trúc này giúp tránh việc Simulator Adapter, Reward Engine và Benchmark Runner tự đọc cấu hình riêng theo những cách khác nhau.

#### 3.2. Simulator Adapter

Simulator Adapter là thành phần kỹ thuật rất quan trọng vì nó hiện thực điểm tiếp xúc giữa PyBatGym với thế giới mô phỏng. Trong phạm vi hiện tại, adapter này làm việc với BatSim, nhưng về mặt giao diện, nó nên được thiết kế như một contract tổng quát cho “simulator backend”.

Adapter chịu trách nhiệm thực hiện các thao tác như: khởi tạo simulator từ workload và platform; reset trạng thái mô phỏng; chạy mô phỏng tới decision point tiếp theo; áp dụng một quyết định scheduling vào simulator; truy xuất trạng thái queue, trạng thái tài nguyên và thông tin thời gian hiện tại; phát hiện điều kiện kết thúc episode. Nói cách khác, nếu simulator là “thế giới scheduling”, thì adapter là người phiên dịch giữa thế giới đó và Environment Core.

Một thiết kế adapter tốt sẽ giúp phần còn lại của hệ thống gần như không cần biết backend đang là BatSim hay một simulator khác. Điều này đặc biệt có giá trị trong báo cáo, vì nó củng cố luận điểm rằng BatSim không phải bản chất của PyBatGym mà chỉ là backend đầu tiên được dùng để hiện thực. Đồng thời, việc tách adapter ra khỏi Environment Core cũng giúp việc kiểm thử dễ hơn: có thể kiểm thử luồng event-driven của adapter mà không cần kéo cả policy loop vào.

#### 3.3. Environment Core

Environment Core là hạt nhân vận hành của PyBatGym. Đây là lớp mà RL framework sẽ tương tác trực tiếp thông qua `reset()` và `step(action)`. Tuy nhiên, vai trò của lớp này không phải là ôm hết logic của hệ thống, mà là điều phối các mô-đun chuyên trách để hoàn thành một transition RL hợp lệ.

Khi `reset()` được gọi, Environment Core lấy cấu hình hiện tại, yêu cầu Simulator Adapter khởi tạo mô phỏng, yêu cầu Episode Manager mở một episode mới, và gọi Observation Encoder để tạo observation ban đầu sau khi simulator được tiến tới decision point đầu tiên. Khi `step(action)` được gọi, Environment Core chuyển action qua Action Processor, yêu cầu Simulator Adapter áp dụng quyết định scheduling, sau đó lấy trạng thái mới, yêu cầu Reward Engine tính reward, yêu cầu Observation Encoder tạo observation kế tiếp, và yêu cầu Episode Manager xác định episode đã kết thúc hay chưa.

Thiết kế này rất quan trọng vì nó giữ cho Environment Core đúng nghĩa là một **orchestrator**. Nếu Environment Core tự tính reward, tự parse state simulator, tự log artifact và tự kiểm tra action, code sẽ nhanh chóng trở thành một khối khó bảo trì. Trong đề tài này, Environment Core phải là điểm tập trung semantics RL, nhưng không phải nơi hiện thực tất cả chi tiết kỹ thuật.

#### 3.4. Observation Encoder

Observation Encoder là mô-đun chuyển trạng thái nội bộ của simulator thành observation có thể đưa cho policy. Đây là một trong các thành phần học thuật quan trọng nhất của hệ thống, vì toàn bộ chất lượng ra quyết định của agent phụ thuộc trực tiếp vào cách trạng thái được trừu tượng hóa.

Trong scheduling simulation, trạng thái thật của hệ thống thường rất lớn và không ổn định: số lượng job trong queue thay đổi liên tục, số job đang chạy thay đổi, trạng thái tài nguyên cũng biến động theo thời gian. Vì vậy, không thể đưa toàn bộ state thô vào policy một cách trực tiếp. PyBatGym phải chuẩn hóa state này thành một observation có kích thước cố định và có nội dung đủ dùng cho ra quyết định.

Cách thiết kế phù hợp cho phiên bản hiện tại là chia observation thành ba nhóm thông tin. Nhóm thứ nhất là **global features**, ví dụ thời gian mô phỏng, số lượng job đang chờ, số lượng job đang chạy, tổng số tài nguyên đang rảnh, mức sử dụng tài nguyên, hoặc waiting time trung bình hiện tại. Nhóm thứ hai là **Top-K queued jobs**, tức chỉ lấy K job đầu theo một quy tắc cố định và biểu diễn một số thuộc tính của chúng như runtime ước lượng, số core yêu cầu, thời gian chờ, tuổi job hoặc các đặc trưng liên quan. Nhóm thứ ba là **resource features**, ví dụ số node rảnh, số core rảnh hoặc mức phân mảnh tài nguyên.

Observation Encoder phải bảo đảm ba tính chất: kích thước đầu ra cố định, nội dung nhất quán theo mọi decision point, và đủ giàu thông tin để policy không bị “mù” trước các yếu tố quan trọng của scheduling. Nếu queue ngắn hơn K, hệ thống phải có quy tắc padding rõ ràng. Nếu dữ liệu thô của simulator có nhiều định dạng khác nhau, Encoder phải chuẩn hóa lại trước khi trả observation ra ngoài.

#### 3.5. Action Processor

Action Processor là mô-đun nhận action từ policy, kiểm tra tính hợp lệ và ánh xạ action đó thành quyết định scheduling thực tế. Trong giải pháp đã chốt, action space nên được giữ đơn giản, ví dụ dạng **Discrete(K)**, trong đó mỗi action tương ứng với việc chọn một job trong Top-K jobs quan sát được.

Thiết kế của Action Processor nên tách thành hai phần. Phần thứ nhất là **Action Validator**, kiểm tra action có nằm trong miền cho phép không, có trỏ vào một job thật hay chỉ là padding, và job đó có đủ điều kiện được cấp phát ở decision point hiện tại hay không. Phần thứ hai là **Action Mapper**, biến action trừu tượng đó thành quyết định cụ thể mà simulator backend hiểu được, ví dụ chọn job nào để dispatch.

Đây là mô-đun đặc biệt quan trọng về mặt giải thích. Một trong những câu hỏi dễ bị hội đồng đào sâu là: “agent thực sự đang điều khiển cái gì?” Nếu Action Processor được thiết kế rõ, sinh viên có thể trả lời rất dứt khoát: agent không điều khiển simulator ở mức thấp; agent chỉ chọn một hành động trừu tượng trong action space; sau đó Action Processor ánh xạ hành động đó thành quyết định scheduling cụ thể trên simulator. Phản biện trước đó cũng đã chạm đúng vào điểm này.

#### 3.6. Reward Engine

Reward Engine chịu trách nhiệm tính reward cho mỗi transition. Về mặt kỹ thuật, reward không nên bị hard-code trực tiếp trong `step()`, mà cần được tách riêng thành mô-đun độc lập. Lý do là reward là thành phần thường xuyên thay đổi hoặc cần điều chỉnh trong nghiên cứu, và nếu trộn trực tiếp vào luồng step thì rất khó kiểm thử, rất khó giải thích và rất khó log decomposition.

Trong phạm vi hiện tại, đề tài chỉ nên dùng một reward function cố định. Reward đó có thể được xây dựng theo hướng kết hợp một số mục tiêu cơ bản như giảm waiting time và tăng utilization. Quan trọng nhất là reward phải có logic giải thích được. Reward Engine nên nhận trạng thái trước và sau transition, hoặc nhận các metric cục bộ sinh ra trong quá trình simulator chạy từ decision point này sang decision point kế tiếp, rồi trả về scalar reward cho agent.

Nếu hệ thống cho phép, Reward Engine cũng nên trả ra **reward decomposition**. Đây là dữ liệu rất mạnh cho việc debug và giải thích, vì nó cho phép chỉ ra reward ở bước này tăng hoặc giảm là vì waiting time thay đổi, utilization thay đổi, hay có thêm job hoàn thành. Với một đề tài framework, khả năng giải thích reward quan trọng hơn việc reward có “hay” đến mức nào.

#### 3.7. Episode Manager

Episode Manager quản lý vòng đời của một episode. Nó giữ trạng thái như episode đã bắt đầu hay chưa, đã kết thúc hay chưa, tổng reward đã tích lũy, số decision point đã đi qua và các metric tổng hợp cuối episode. Trong môi trường event-driven, Episode Manager đặc biệt quan trọng vì môi trường không tiến theo đồng hồ cố định; do đó, việc biết khi nào nên dừng và trạng thái hiện tại của một episode phải được kiểm soát rất chặt.

Episode Manager nhận thông tin từ Environment Core và Simulator Adapter để xác định điều kiện kết thúc. Điều kiện chuẩn cho phiên bản hiện tại nên là: toàn bộ job trong workload đã hoàn tất và không còn job chờ hoặc job chạy; hoặc đạt ngưỡng cắt episode theo cấu hình nếu có. Mô-đun này cũng nên là nơi tích lũy metric cuối episode để Logging Manager và Benchmark Runner có thể sử dụng.

#### 3.8. Logging and Artifact Manager

Đây là mô-đun phục vụ trực tiếp cho evaluation và reproducibility. Nó cần ghi lại ít nhất ba lớp thông tin. Lớp thứ nhất là **experiment metadata**, gồm config, seed, workload, backend đang dùng, policy đang dùng. Lớp thứ hai là **step-level logs**, gồm decision point index, action, reward, action mask nếu có, và một số đặc trưng tóm tắt của queue hoặc resource state. Lớp thứ ba là **episode-level metrics**, gồm tổng reward, waiting time trung bình, slowdown trung bình, utilization hoặc các chỉ số chính khác.

Artifact Manager cũng cần xuất ra các file mà báo cáo có thể dùng trực tiếp, ví dụ CSV cho episode metrics, file cấu hình, seed log và bảng so sánh giữa các policy. Theo định hướng của slide deck, artifact là một phần của giá trị hệ thống chứ không chỉ là phần thêm sau cùng. Điều đó có nghĩa là mô-đun này phải được xem như một phần của kiến trúc lõi của giải pháp nghiên cứu.

#### 3.9. Benchmark Runner

Benchmark Runner không nhất thiết phải phức tạp, nhưng phải tồn tại. Nó là mô-đun điều phối việc chạy nhiều policy trên cùng workload và cùng config, sau đó thu metric và tổng hợp thành bảng so sánh. Trong scope của 1 sinh viên, Benchmark Runner chỉ cần đủ để chạy tối thiểu một heuristic baseline và một policy đơn giản, rồi xuất comparison table.

Vai trò của Benchmark Runner rất lớn về mặt trình bày. Nó chứng minh rằng PyBatGym không chỉ “mở được simulator và chạy được agent”, mà còn hỗ trợ được một workflow nghiên cứu tối thiểu. Điều này cũng bám rất sát tiêu chí đánh giá đã được nêu trong slide: so sánh heuristic scheduler và một policy đơn giản trên cùng workload.

---

### 4. Thiết kế chi tiết luồng xử lý kỹ thuật

#### 4.1. Luồng reset

Khi RL framework hoặc script đánh giá gọi `reset()`, Environment Core trước hết lấy cấu hình hiện tại từ Configuration Manager. Sau đó, nó yêu cầu Simulator Adapter reset backend mô phỏng với workload, platform và seed tương ứng. Episode Manager được làm sạch trạng thái. Logging Manager tạo một phiên log mới.

Tiếp theo, Simulator Adapter tiến mô phỏng tới decision point đầu tiên. Trạng thái thô của simulator tại thời điểm này được chuyển cho Observation Encoder để tạo observation đầu tiên. Observation được trả về cho caller như output của `reset()`. Toàn bộ bước này phải diễn ra trơn tru và nhất quán, vì nó là điểm khởi đầu của mọi episode.

#### 4.2. Luồng step

Khi caller gửi `step(action)`, Environment Core nhận action và chuyển sang Action Processor. Nếu action hợp lệ, nó được ánh xạ thành một quyết định scheduling cụ thể và gửi xuống Simulator Adapter. Adapter áp dụng quyết định vào simulator rồi chạy tiếp cho tới decision point kế tiếp hoặc tới khi episode kết thúc.

Trạng thái mới nhận được từ simulator được gửi sang Reward Engine để tính reward. Sau đó, Observation Encoder xây dựng observation mới từ trạng thái đó. Episode Manager cập nhật tổng reward, decision count và kiểm tra điều kiện kết thúc. Logging Manager ghi lại transition vừa diễn ra. Cuối cùng, Environment Core trả ra observation mới, reward, done và info cho caller.

Cách tổ chức này cho thấy rõ semantics event-driven: mỗi lần `step()` không đại diện cho một tick thời gian, mà đại diện cho một vòng quyết định hoàn chỉnh giữa hai decision points.

---

### 5. Thiết kế dữ liệu chính của hệ thống

Ở mức dữ liệu, hệ thống cần một số object logic rõ ràng.

- **ExperimentConfig** là cấu trúc chứa toàn bộ thông tin cấu hình của một lần chạy. Nó là đầu vào chung cho hầu hết các mô-đun.

- **SimulationState** là trạng thái thô nhận từ simulator adapter, bao gồm thông tin queue, running jobs, resources, simulation time và metadata liên quan.

- **Observation** là trạng thái đã được mã hóa, có kích thước cố định và được gửi cho policy.

- **ActionContext** là ngữ cảnh xử lý action tại decision point hiện tại, bao gồm Top-K jobs, action mask và các thông tin cần cho Action Validator và Action Mapper.

- **TransitionResult** là kết quả của một lần step, gồm observation mới, reward, done/info và các metric phụ.

- **EpisodeSummary** là tổng hợp cuối episode, được dùng cho artifact export và benchmark.

Nếu báo cáo đi sâu hơn, sinh viên có thể chuyển các object logic này thành class diagram hoặc dataclass definitions. Điều quan trọng là phải cho thấy hệ thống không chỉ có “các hàm”, mà có cả một mô hình dữ liệu nội bộ nhất quán.

---

### 6. Thiết kế kỹ thuật tối thiểu cho nhóm

Vì giờ chỉ còn 1 sinh viên cho đồ án, việc thiết kế kỹ thuật phải được chốt ở mức **tối thiểu nhưng có giá trị chứng minh**. Điều này nghĩa là không nên tham vọng quá nhiều abstraction hoặc extension points. Kiến trúc nên đủ đẹp để hội đồng thấy tư duy hệ thống, nhưng code phải tập trung vào một vertical slice thật sự chạy được.

Do đó, một số quyết định kỹ thuật nên được chốt rất sớm và không thay đổi nữa:

- chỉ có một backend adapter là BatSim;
    
- chỉ có một observation schema;
    
- chỉ có một action schema;
    
- chỉ có một reward function;
    
- chỉ có một benchmark pipeline đơn giản;
    
- chỉ xuất những artifact cần thiết nhất.


Đây là điểm cực kỳ quan trọng. Trong đồ án kiểu này, rủi ro lớn nhất không phải là “thiết kế chưa đẹp”, mà là “thiết kế quá đẹp nhưng không chạy hết luồng”. Vì vậy, kỹ thuật phải phục vụ khả năng hoàn thành chứ không chỉ phục vụ tham vọng kiến trúc.

---

### 7. Kết luận 

Về bản chất, thiết kế kỹ thuật của PyBatGym nhằm hiện thực một event-driven RL environment cho scheduling simulation theo cách có thể tích hợp với simulator backend, có thể dùng bởi RL framework, và có thể phục vụ benchmark nghiên cứu ở mức tối thiểu. Kiến trúc của hệ thống được tổ chức theo hướng tách mô-đun rõ ràng, trong đó Environment Core điều phối các thành phần cấu hình, adapter, observation, action, reward, episode, logging và benchmark. Cách thiết kế này cho phép đề tài giữ được chiều sâu học thuật ở phần mô hình hóa MDP và semantics của environment, đồng thời vẫn bảo đảm tính khả thi trong bối cảnh chỉ còn một sinh viên thực hiện. BatSim xuất hiện trong hệ thống với vai trò backend đầu tiên để kiểm chứng, chứ không phải lõi bản chất của giải pháp. Chính điểm đó giúp PyBatGym giữ được định vị như một framework event-driven scheduling environment, thay vì chỉ bị xem là một wrapper tích hợp kỹ thuật.
