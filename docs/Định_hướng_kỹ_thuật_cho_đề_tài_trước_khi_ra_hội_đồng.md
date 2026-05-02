### 1. Mục tiêu cuối cùng của đề tài ở thời điểm ra hội đồng

Ở giai đoạn cuối kỳ, mục tiêu của đề tài **không còn là cố gắng hiện thực một framework nghiên cứu hoàn chỉnh theo nghĩa đầy đủ nhất**, mà là **chứng minh được một lõi giải pháp đúng, chạy được, giải thích được và đánh giá được**. Với bối cảnh chỉ còn một sinh viên thực hiện, giá trị lớn nhất của đề tài sẽ không nằm ở số lượng chức năng, mà nằm ở việc sinh viên có thể trình bày được một hệ thống có kiến trúc rõ ràng, có pipeline end-to-end, có tiêu chí đánh giá cụ thể, và có bằng chứng kỹ thuật đủ mạnh để bảo vệ trước hội đồng.

Vì vậy, mục tiêu cuối cùng của đề tài nên được chốt như sau:

==Đề tài xây dựng một **prototype PyBatGym tối thiểu nhưng hoàn chỉnh theo chiều dọc**, có khả năng mô hình hóa bài toán scheduling simulation theo hướng event-driven thành một môi trường tương thích Gymnasium, sử dụng **BatSim như backend đầu tiên để kiểm chứng**, hỗ trợ chạy được một episode hoàn chỉnh, hỗ trợ so sánh một heuristic baseline với một policy đơn giản, và hỗ trợ lưu lại cấu hình – seed – metric để bảo đảm tính tái lập ở mức cơ bản. Đây là cách phát biểu phù hợp với mục tiêu ban đầu của đề tài, đồng thời đáp ứng được yêu cầu hội đồng từng đặt ra về tính khả thi, về logic MDP, và về câu hỏi agent quan sát – hành động – tác động lên hệ thống như thế nào.==

Nói cách khác, đến cuối kỳ sinh viên không cần chứng minh rằng đã xây xong một “platform lớn”, mà cần chứng minh bốn điều: **hệ thống được thiết kế đúng hướng, hệ thống chạy được, hệ thống benchmark được ở mức tối thiểu, và hệ thống có thể được giải thích một cách chặt chẽ**.

---

### 2. Cách trình bày đề tài trước hội đồng

Khi bảo vệ, sinh viên phải trình bày đề tài theo một cách nhất quán, tránh để hội đồng hiểu sai rằng đây chỉ là một lớp bọc kỹ thuật quanh BatSim. Cách định vị hợp lý nên dùng là:

PyBatGym là một **framework môi trường học tăng cường cho scheduling simulation theo hướng event-driven**. Hệ thống chuẩn hóa tương tác giữa policy và simulator backend thông qua giao diện kiểu Gymnasium, cho phép mô hình hóa observation, action, reward và episode semantics trong bối cảnh scheduling. Trong phạm vi hiện tại, BatSim được dùng làm backend đầu tiên để hiện thực và kiểm chứng giải pháp. Tên gọi “PyBatGym” giữ lại dấu vết cảm hứng từ hướng tiếp cận event-driven scheduling simulation, chứ không hàm ý rằng hệ thống chỉ phục vụ riêng cho BatSim.

Điểm này rất quan trọng. Nếu sinh viên trình bày sai trọng tâm, toàn bộ đề tài dễ bị kéo về một nhận xét bất lợi, ví dụ: “đây chỉ là wrapper cho BatSim”, hoặc “tại sao phải đặt tên riêng trong khi chỉ kết nối RL vào BatSim”. Vì vậy, phần mở đầu khi thuyết trình phải nhấn mạnh ba tầng rõ ràng:

- **Bản chất hệ thống**: framework environment cho event-driven scheduling simulation.
- **Hiện thực hiện tại**: BatSim là backend đầu tiên được tích hợp.
- **Phạm vi đồ án**: chỉ kiểm chứng prototype tối thiểu, chưa nhằm hỗ trợ nhiều simulator.

Nếu làm được điều này, hội đồng sẽ dễ nhìn thấy giá trị của đề tài ở tầng **mô hình hóa và chuẩn hóa môi trường RL**, thay vì chỉ nhìn nó như một bài tích hợp công cụ.

---

### 3. Scope hoàn thiện cho đồ án tốt nghiệp

Với nguồn lực hiện tại, scope phải được rút xuống mức **đủ sâu theo một luồng hoàn chỉnh**, thay vì cố rộng theo nhiều tính năng. Cách chốt tốt nhất là coi đề tài như một **vertical slice research prototype**.

Scope bắt buộc phải hoàn thành gồm các phần sau.

- Thứ nhất, hệ thống chỉ hỗ trợ **một simulator backend duy nhất là BatSim**. Dù kiến trúc có thể được trình bày theo tư duy backend-agnostic, implementation chỉ cần hiện thực một adapter cho BatSim. Không mở rộng sang simulator thứ hai, không cố xây abstraction quá tổng quát đến mức tốn công nhưng không có bằng chứng chạy thực.

- Thứ hai, hệ thống chỉ hỗ trợ **một environment variant duy nhất**. Điều này có nghĩa là observation schema, action schema và reward function phải được chốt một lần và không mở thêm nhiều mode để người dùng lựa chọn. Một environment chạy được, rõ semantics và có thể giải thích được sẽ mạnh hơn nhiều so với nhiều lựa chọn nửa vời.

- Thứ ba, action space nên được chốt ở mức đơn giản nhưng khả thi, ví dụ dạng **Discrete(K)** với ý nghĩa chọn một job trong **Top-K jobs** quan sát được. Thiết kế này vừa giúp kiểm soát độ phức tạp, vừa dễ giải thích với hội đồng, vừa phù hợp với các RL algorithm phổ biến. Đây cũng là hướng đã được gợi ra trong slide của đề tài.

- Thứ tư, reward function phải chỉ có **một phiên bản cố định**, có thể đơn giản hóa thành một công thức ưu tiên giảm waiting time và cải thiện utilization, miễn là sinh viên giải thích rõ được logic của reward và có thể log để đối chiếu trong lúc chạy. Không nên cố thử nhiều reward modes trong giai đoạn này.

- Thứ năm, policy benchmark chỉ cần gồm **một heuristic baseline** và **một policy đơn giản**. Policy đơn giản ở đây không nhất thiết phải là RL policy mạnh; có thể là random policy, simple rule policy hoặc một agent RL rất nhẹ nếu kịp. Điều cốt lõi là chứng minh framework có thể đặt hai chính sách khác bản chất vào cùng một pipeline đánh giá.

- Thứ sáu, hệ thống phải lưu được **config – seed – episode metrics cơ bản**. Không cần xây dashboard lớn, nhưng nhất định phải có logging đủ để chứng minh reproducibility và đủ để trình bày evaluation.

Những phần **không nên để trong scope chính** gồm: hỗ trợ nhiều backend simulator, nhiều reward modes, nhiều observation schemas, nhiều baseline heuristic, training RL dài hơi để lấy số đẹp, UI dashboard phức tạp, tối ưu hiệu năng framework, hay khẳng định khả năng plug-and-play với mọi simulator. Với 1 sinh viên, các phần đó chỉ nên được nêu như hướng mở rộng sau này, không phải cam kết phải hoàn thành.

---

### 4. Các kỹ thuật cốt lõi của giải pháp cần đảm bảo

Trong toàn bộ đề tài, có bốn hạt nhân kỹ thuật mà sinh viên bắt buộc phải nắm và trình bày vững.

#### 4.1. Event-driven scheduling environment

Đây là linh hồn của PyBatGym. Sinh viên phải giải thích được rằng trong scheduling simulation, agent không tương tác ở mọi time-step đều nhau, mà chỉ ở những **decision points** có ý nghĩa scheduling, ví dụ khi có job mới đến, khi job hoàn thành, hoặc khi tài nguyên được giải phóng đủ để hình thành một quyết định mới. Chính vì vậy, mỗi lần `step()` trong môi trường không đại diện cho một đơn vị thời gian cố định, mà đại diện cho **một cơ hội ra quyết định**. Cách tiếp cận này làm cho environment sát hơn với bài toán scheduling thật, thay vì nhồi agent vào một time loop cơ học. Đây cũng là một điểm đã được nêu rất rõ trong slide deck.

#### 4.2. Mô hình hóa observation – action – reward

Đây là phần hội đồng rất dễ hỏi sâu. Sinh viên phải nói rõ: trạng thái nội bộ đầy đủ của simulator là quá lớn và không thể đưa trực tiếp vào agent; vì vậy PyBatGym cần một **Observation Encoder** để rút gọn trạng thái thành observation có kích thước cố định. Observation tối thiểu nên gồm ba nhóm: đặc trưng toàn cục của hệ thống, đặc trưng Top-K jobs đang chờ, và đặc trưng tài nguyên hiện tại.

Action được biểu diễn thành việc chọn một job trong Top-K. Reward phản ánh chất lượng scheduling sau transition, chẳng hạn hướng tới giảm waiting time và sử dụng tài nguyên hợp lý hơn. Ở đây, điều quan trọng không phải reward có “tối ưu” hay không, mà là sinh viên phải chỉ ra được **vì sao reward đó hợp lý, được tính ở đâu, và có thể kiểm chứng bằng log như thế nào**.

#### 4.3. Gym-compatible interaction loop

Sinh viên phải chứng minh được rằng hệ thống không chỉ là mô hình nội bộ, mà thực sự có một interaction loop tương thích với Gymnasium: `reset()` khởi tạo episode và trả observation đầu tiên; `step(action)` nhận action, ánh xạ sang quyết định scheduling, điều khiển simulator chạy tới decision point tiếp theo, tính reward, sinh observation mới và trả cờ kết thúc. Đây là bằng chứng mạnh nhất để nói rằng đề tài đã đạt được lớp environment chứ không dừng ở mức thiết kế ý tưởng.

#### 4.4. Benchmark và reproducibility

Phần này không được xem là “phụ”. Nếu không benchmark được và không lưu được config/seed/log, thì đề tài rất dễ bị xem là chỉ có demo kỹ thuật. Vì vậy, sinh viên phải có một pipeline tối thiểu để chạy hai policy trên cùng điều kiện và xuất được metric, đồng thời giữ lại seed và config để có thể chạy lại. Theo slide, đây là một phần trong tiêu chí đánh giá cốt lõi của hệ thống.

---

### 5. Deliverable kỹ thuật bắt buộc trước khi ra hội đồng

Để tránh mơ hồ, giai đoạn cuối kỳ cần được quản lý theo **deliverable kỹ thuật cụ thể**, không theo các mô tả chung chung kiểu “hoàn thiện hệ thống”.

Sinh viên bắt buộc phải có các deliverable sau.

- Thứ nhất là **sơ đồ kiến trúc cuối cùng của hệ thống**, thể hiện rõ các mô-đun chính: configuration manager, simulator adapter, environment core, observation encoder, action processor, reward engine, episode manager, logger và benchmark runner. Sơ đồ này phải nhất quán với code thực tế, không được vẽ quá đẹp rồi code lại không có.

- Thứ hai là **mô tả chính thức observation, action, reward và episode**. Đây nên là một bảng hoặc một mục riêng trong báo cáo. Hội đồng phải đọc vào là hiểu environment đang định nghĩa bài toán thế nào. Nếu phần này còn mơ hồ, đề tài sẽ yếu ngay từ lõi.

- Thứ ba là **code chạy end-to-end một episode**. Đây là bằng chứng kỹ thuật quan trọng nhất. Phải có script hoặc notebook cho thấy có thể khởi tạo environment, chạy reset, step nhiều lần, và kết thúc episode thành công.

- Thứ tư là **một baseline heuristic**. Baseline không cần mạnh, nhưng phải rõ logic. Ví dụ có thể là FCFS hoặc một policy heuristic đơn giản phù hợp với queue hiện tại. Điều quan trọng là baseline phải chạy qua cùng pipeline của environment.

- Thứ năm là **một policy đơn giản thứ hai** để benchmark với heuristic. Có thể là random policy, fixed-order policy, hoặc một RL policy sơ cấp nếu đủ thời gian. Điều cốt lõi là hai policy phải được đánh giá trong cùng điều kiện.

- Thứ sáu là **artifact đánh giá**, tối thiểu gồm: config file, seed log, episode metrics dạng CSV, và một bảng so sánh giữa hai policy.

- Thứ bảy là **một log debug ở decision point**. Đây là bằng chứng rất mạnh trước hội đồng, vì nó cho thấy framework có thể giải thích hành vi agent thay vì là hộp đen. Phản biện trước đó từng chạm đúng vào vấn đề này.

Nếu một trong các deliverable trên còn thiếu, phần trình bày cuối kỳ sẽ rất dễ bị hẫng.

---

### 6. Hướng tiếp cận kỹ thuật của đồ án

Ở thời điểm này, sinh viên không nên tiếp tục thử quá nhiều hướng nữa. Cần chốt một hướng tiếp cận duy nhất và đi đến cùng. Hướng đó nên là:

PyBatGym được xây dựng như một **event-driven Gym-compatible environment prototype** cho scheduling simulation. BatSim được tích hợp như backend đầu tiên thông qua một adapter. Observation được mã hóa từ trạng thái simulator thành vector cố định. Action là chọn job từ Top-K. Reward được tính theo một công thức cố định phản ánh chất lượng scheduling. Episode được gắn với một workload trace. Hệ thống cho phép chạy một baseline heuristic và một policy đơn giản trên cùng cấu hình, đồng thời lưu config, seed và metric đầu ra.

Cách tiếp cận này có ba ưu điểm. Thứ nhất, nó đủ nhỏ để 1 sinh viên hoàn thành. Thứ hai, nó vẫn giữ được bản sắc nghiên cứu của đề tài. Thứ ba, nó trả lời đúng ba câu hỏi lớn của hội đồng: hệ thống có kiến trúc gì, agent tương tác với scheduling như thế nào, và kết quả được đánh giá ra sao.

---

### 7. Những hiểu lầm cần tránh khi trình bày

Trong giai đoạn cuối, sinh viên phải đặc biệt tránh một số cách nói dễ làm đề tài bị đánh giá thấp.

- Không nên nói rằng đề tài “xây dựng framework tổng quát cho mọi simulator”. Vì implementation hiện tại chỉ có BatSim, nếu phát biểu quá lớn sẽ bị hỏi ngược rất mạnh.
- Không nên nói rằng “mục tiêu là xây dựng scheduler RL tốt hơn heuristic”. Vì đây không phải tiêu chí thành bại chính của framework. Nếu RL chưa mạnh, đề tài vẫn có thể đạt nếu framework chạy đúng và benchmark được.
- Không nên nói PyBatGym là “wrapper cho BatSim”. Điều này làm giảm ngay giá trị của phần mô hình hóa environment. Thay vào đó, phải nói BatSim là backend đầu tiên được tích hợp để kiểm chứng.
- Không nên nói quá nhiều về tương lai mà không có bằng chứng hiện tại. Những hướng như multi-backend, multi-agent, nhiều reward modes chỉ nên để ở phần future work.

---

### 8. Cách đánh giá cuối kỳ 

Đánh giá cuối kỳ nên xoay quanh ba nhóm tiêu chí.

- Nhóm thứ nhất là **tính đúng đắn kỹ thuật của environment**. Hệ thống phải chạy được một episode end-to-end, có observation hợp lệ, action xử lý đúng, reward được tính, episode kết thúc đúng điều kiện. Đây là tầng nền tảng nhất.

- Nhóm thứ hai là **khả năng benchmark tối thiểu**. Hệ thống phải so sánh được heuristic và policy đơn giản trên cùng workload, cùng config, cùng seed hoặc tập seed xác định trước. Ở đây không yêu cầu policy RL phải thắng; điều cần là benchmark pipeline vận hành đúng.

- Nhóm thứ ba là **khả năng tái lập và giải thích**. Hệ thống phải lưu được config, seed, metric và có log đủ để phân tích một vài decision point tiêu biểu. Đây là điểm giúp đề tài thoát khỏi mức demo và tiến gần hơn tới một research framework tối thiểu.

---

### 9. Checklist cuối cùng trước khi ra hội đồng

Sinh viên chỉ nên xem là “sẵn sàng bảo vệ” khi trả lời được “có” cho toàn bộ các câu hỏi sau:

- [ ] Tôi đã chốt một mô tả nhất quán về PyBatGym chưa?
- [ ] Tôi đã giải thích rõ vì sao PyBatGym không lấy BatSim làm core chưa?
- [ ] Tôi đã có sơ đồ kiến trúc khớp với code hiện tại chưa?
- [ ] Tôi đã chốt observation, action, reward, episode chưa?
- [ ] Tôi đã có code chạy end-to-end một episode chưa?
- [ ] Tôi đã có ít nhất một baseline heuristic chưa?
- [ ] Tôi đã có policy thứ hai để benchmark chưa?
- [ ] Tôi đã có artifact gồm config, seed, metrics và comparison table chưa?
- [ ] Tôi đã có một ví dụ log decision point để giải thích hành vi policy chưa?
- [ ] Tôi đã có cách giải thích giới hạn của đề tài mà không bị xem là né tránh chưa?

Nếu còn bất kỳ câu nào chưa có, thì ưu tiên của sinh viên phải là **đóng lỗ hổng đó**, không phải thêm tính năng mới.