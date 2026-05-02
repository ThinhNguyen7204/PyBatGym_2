
## Đánh giá và thử nghiệm

### 1. Mục tiêu của phần đánh giá

Phần đánh giá và thử nghiệm của đề tài không nhằm chứng minh rằng một policy học tăng cường cụ thể đã vượt trội so với mọi heuristic trong bài toán lập lịch HPC. Thay vào đó, mục tiêu chính là kiểm chứng rằng **PyBatGym đã được hiện thực như một môi trường RL có thể vận hành được, có thể dùng để benchmark ở mức tối thiểu, và có thể tái lập thí nghiệm**. Cách tiếp cận này phù hợp với bản chất của đề tài là xây dựng một framework môi trường cho scheduling simulation, thay vì tập trung vào việc phát triển hoặc tối ưu một thuật toán RL mới.

Từ góc nhìn đó, phần đánh giá cần trả lời ba câu hỏi chính. Thứ nhất, môi trường có thực sự chạy được end-to-end theo semantics của Gymnasium hay không. Thứ hai, môi trường có cho phép đặt nhiều chính sách lập lịch vào cùng một pipeline đánh giá để so sánh hay không. Thứ ba, hệ thống có lưu được đủ cấu hình, seed và artifact để kết quả có thể được tái dựng và phân tích lại hay không. Ba câu hỏi này bám sát với bộ tiêu chí đánh giá mà đề tài đã xác lập: chạy được một episode đầy đủ, so sánh được heuristic với policy đơn giản, và kiểm tra reproducibility thông qua seed cùng config.

---

### 2. Nguyên tắc đánh giá

Việc đánh giá được thực hiện theo nguyên tắc ưu tiên **tính đúng đắn và tính hoàn chỉnh của pipeline kỹ thuật** hơn là mức độ “đẹp” của kết quả tối ưu. Điều này đặc biệt quan trọng trong bối cảnh chỉ còn một sinh viên thực hiện đề tài. Nếu lấy hiệu năng cuối cùng của một policy RL làm trung tâm, đề tài sẽ bị kéo sang hướng nghiên cứu thuật toán, trong khi phần lõi cần chứng minh lại là framework environment. Vì vậy, mọi thử nghiệm trong phần này phải phục vụ cho việc chứng minh rằng các thành phần kỹ thuật của hệ thống đã phối hợp được với nhau trong một quy trình đầy đủ và có ý nghĩa.

Một nguyên tắc khác là các thử nghiệm phải được xây dựng trên **một vertical slice nhất quán**, tức cùng một environment variant, cùng một observation schema, cùng một action schema, cùng một reward function và cùng một backend simulator. Điều này giúp giảm nhiễu trong quá trình đánh giá và làm cho kết quả dễ diễn giải hơn. Với phiên bản hiện tại, BatSim được dùng làm backend đầu tiên để kiểm chứng, nhưng trọng tâm đánh giá không phải là BatSim tự thân, mà là khả năng của PyBatGym trong việc điều phối interaction loop giữa policy và scheduling simulator theo hướng event-driven.

---

### 3. Câu hỏi nghiên cứu và câu hỏi kỹ thuật cần kiểm chứng

Phần đánh giá của đề tài có thể được tổ chức quanh một số câu hỏi kiểm chứng cụ thể.

Câu hỏi thứ nhất là: **PyBatGym có vận hành được như một Gym-compatible environment cho scheduling simulation hay không?** Đây là câu hỏi nền tảng nhất, vì nếu không trả lời được, mọi đánh giá tiếp theo đều mất ý nghĩa. Câu hỏi này tương ứng với việc kiểm tra reset-step loop, episode termination, observation generation, action handling và reward computation.

Câu hỏi thứ hai là: **PyBatGym có hỗ trợ được benchmark tối thiểu giữa các chính sách lập lịch khác nhau hay không?** Ở đây benchmark tối thiểu được hiểu là có thể đặt ít nhất một heuristic baseline và một policy đơn giản vào cùng workload, cùng config, rồi thu được các metric đầu ra có thể đối chiếu.

Câu hỏi thứ ba là: **PyBatGym có hỗ trợ reproducibility ở mức cơ bản hay không?** Điều này được kiểm tra thông qua việc cố định seed, lưu cấu hình và chạy lại cùng một thí nghiệm để quan sát độ nhất quán của kết quả.

Câu hỏi thứ tư, mang tính bổ sung nhưng rất quan trọng khi ra hội đồng, là: **PyBatGym có cung cấp đủ log để giải thích decision point hay không?** Phản biện giai đoạn trước đã cho thấy hội đồng rất quan tâm đến việc agent quan sát gì và hành động như thế nào. Do đó, nếu framework ghi được log ở decision point thì phần bảo vệ sẽ thuyết phục hơn nhiều.

---

### 4. Thiết lập thử nghiệm

#### 4.1. Thiết lập môi trường

Toàn bộ thử nghiệm được thực hiện trên một môi trường PyBatGym đã được chốt với một observation schema duy nhất, một action schema duy nhất và một reward function duy nhất. Điều này giúp giữ cho đánh giá tập trung vào việc kiểm chứng framework, thay vì phân tán sang việc so sánh nhiều biến thể thiết kế environment. Environment vận hành theo cơ chế event-driven, trong đó mỗi lần `step()` tương ứng với một decision point scheduling có ý nghĩa, thay vì một time-step cố định. Đây là điểm then chốt của hệ thống và cũng là phần cần được kiểm chứng trong toàn bộ evaluation.

Backend simulator được sử dụng trong các thử nghiệm là BatSim, thông qua một adapter đã được hiện thực trong PyBatGym. Trong báo cáo, cần nhấn mạnh rằng BatSim được sử dụng như backend thực để kiểm chứng vòng lặp kỹ thuật, chứ không phải để định nghĩa bản chất hệ thống.

#### 4.2. Thiết lập đầu vào

Đầu vào của mỗi thí nghiệm gồm workload trace, platform description, file cấu hình experiment và seed. Để tránh làm phức tạp hóa phần đánh giá, nên sử dụng một tập workload đủ đơn giản để minh họa rõ luồng vận hành, thay vì cố gắng dùng quá nhiều trace phức tạp. Mục tiêu ở đây không phải là bao phủ toàn bộ phổ workload của HPC, mà là tạo ra một điều kiện đủ để framework chứng minh được giá trị của mình.

Cùng với workload và platform, seed phải được cố định và lưu lại trong artifact của từng lần chạy. Điều này cần được áp dụng cho cả heuristic baseline và policy thứ hai, để bảo đảm benchmark có thể được lặp lại theo cùng điều kiện. Theo định hướng của đề tài, config file và random seed là một phần của artifact cần có.

#### 4.3. Chính sách được dùng trong thử nghiệm

Với mục tiêu benchmark tối thiểu, thử nghiệm nên sử dụng hai loại policy. Chính sách thứ nhất là một **heuristic baseline**, ví dụ một heuristic đơn giản và dễ giải thích như FCFS hoặc một quy tắc dispatch đơn giản tương tự. Chính sách thứ hai là một **policy đơn giản** để chứng minh rằng PyBatGym có thể hỗ trợ hơn một loại quyết định. Policy thứ hai không bắt buộc phải là một RL policy mạnh; có thể là random policy, fixed-order policy hoặc một agent rất nhẹ nếu việc tích hợp đã đủ sẵn sàng.

Cách chọn như vậy có lợi ở chỗ nó giữ benchmark ở mức tối thiểu nhưng vẫn đủ ý nghĩa. Hệ thống không bị áp lực phải chứng minh chất lượng tối ưu của RL, nhưng vẫn chứng minh được rằng pipeline của framework chấp nhận nhiều loại policy khác nhau và đánh giá chúng trong cùng điều kiện. Đây là đúng tinh thần của tiêu chí “compare heuristic scheduler and a simple policy on the same workload” mà slide deck đã nêu.

---

### 5. Các nhóm thử nghiệm chính

#### 5.1. Thử nghiệm kiểm tra luồng end-to-end

Nhóm thử nghiệm đầu tiên nhằm kiểm tra xem environment có chạy được một episode hoàn chỉnh hay không. Trong thử nghiệm này, hệ thống được khởi tạo bằng `reset()`, sau đó một policy đơn giản sẽ sinh action liên tục cho đến khi episode kết thúc. Trong suốt quá trình này, cần quan sát xem environment có trả về observation hợp lệ, có xử lý action đúng, có tính reward và có dừng đúng điều kiện hay không.

Kết quả mong đợi của nhóm thử nghiệm này không phải là một bảng số đẹp, mà là bằng chứng kỹ thuật cho thấy reset-step loop đã hoạt động trọn vẹn. Một log đầy đủ của một episode tiêu biểu, trong đó có observation đầu, một số transition trung gian và trạng thái kết thúc cuối episode, sẽ là minh chứng quan trọng nhất cho mục này. Đây cũng là tiêu chí nền tảng đầu tiên trong bộ tiêu chí đánh giá của đề tài.

#### 5.2. Thử nghiệm kiểm tra observation, action và reward

Nhóm thử nghiệm thứ hai tập trung vào từng thành phần kỹ thuật của environment. Trước hết, observation được kiểm tra tại nhiều decision point khác nhau để bảo đảm kích thước đầu ra là cố định, padding hoạt động đúng và các đặc trưng quan trọng của hệ thống được phản ánh nhất quán. Tiếp theo, action handling được kiểm tra bằng cách quan sát action mask, action được chọn, action mapping và trạng thái simulator sau khi áp dụng action. Cuối cùng, reward được kiểm tra thông qua việc đối chiếu các giá trị reward với biến động thực trong queue và resource state.

Phần thử nghiệm này có ý nghĩa rất lớn khi trình bày với hội đồng, vì nó cho thấy PyBatGym không chỉ “chạy được”, mà còn có semantics được kiểm soát. Đây chính là chỗ để trả lời các câu hỏi kiểu: agent nhìn thấy gì, action đại diện cho cái gì, reward được tính tại đâu và vì sao transition này được thưởng hoặc phạt. Phản biện trước đó đã cho thấy đây là tầng mà nhóm cần làm rõ.

#### 5.3. Thử nghiệm benchmark tối thiểu giữa các policy

Nhóm thử nghiệm thứ ba là benchmark tối thiểu giữa heuristic baseline và policy thứ hai. Hai policy phải được chạy trên cùng workload, cùng platform, cùng config và cùng seed hoặc cùng tập seed cố định. Sau mỗi lần chạy, hệ thống thu các metric chính như total reward, average waiting time, average slowdown, utilization và throughput nếu đã được hỗ trợ trong environment.

Điều quan trọng cần nhấn mạnh là mục tiêu của nhóm thử nghiệm này **không phải chứng minh policy thứ hai vượt heuristic**, mà là chứng minh PyBatGym có thể dùng như một benchmark platform tối thiểu. Chỉ cần hệ thống chạy được hai policy trong cùng pipeline và xuất được comparison table có ý nghĩa, thì tiêu chí benchmark của framework đã được đáp ứng. Điều này hoàn toàn nhất quán với định hướng ban đầu của đề tài.

#### 5.4. Thử nghiệm kiểm tra reproducibility

Nhóm thử nghiệm thứ tư kiểm tra tính tái lập. Với một cấu hình thí nghiệm cố định, workload cố định và seed cố định, hệ thống được chạy lại nhiều lần để quan sát sự nhất quán của kết quả. Nếu framework được thiết kế đúng, các metric cuối episode và chuỗi hành vi chính phải có mức độ ổn định cao giữa các lần chạy cùng điều kiện.

Đây là một thử nghiệm rất quan trọng về mặt nghiên cứu, vì nó cho thấy PyBatGym không chỉ là một prototype demo ngẫu hứng mà đã hỗ trợ được workflow thí nghiệm có thể tái dựng. Theo slide deck, reproducibility thông qua seed và config là một trong các tiêu chí đánh giá trực tiếp của hệ thống.

#### 5.5. Thử nghiệm kiểm tra khả năng debug decision point

Nhóm thử nghiệm cuối cùng mang tính chất hỗ trợ nhưng rất giá trị: kiểm tra khả năng debug ở decision point. Trong thử nghiệm này, một số decision point tiêu biểu trong episode được trích ra và đối chiếu với log chi tiết: observation tóm tắt, queue snapshot, action hợp lệ, action được chọn, reward decomposition và trạng thái sau khi áp dụng action.

Nếu có thể dùng log để tái dựng lại một vài quyết định của policy, sinh viên sẽ có lợi thế rất lớn khi bảo vệ. Hội đồng thường không chỉ nhìn vào metric cuối cùng, mà còn muốn thấy framework có minh bạch hay không. Việc có log decision point sẽ cho phép chứng minh rằng PyBatGym không phải hộp đen và có thể hỗ trợ phân tích hành vi agent. Phản biện trước đó đã ngầm yêu cầu đúng điều này.

---

### 6. Chỉ số đánh giá

Các chỉ số đánh giá trong đề tài nên được chọn theo hướng phục vụ cả **kiểm chứng kỹ thuật** lẫn **đánh giá hành vi scheduling**.

Ở tầng kiểm chứng kỹ thuật, các chỉ số cần có gồm: số episode chạy thành công; số lỗi runtime trong reset-step loop; số transition hợp lệ; tỷ lệ action hợp lệ được xử lý thành công; số lần vi phạm schema observation; số lần episode kết thúc sai hoặc không kết thúc đúng điều kiện. Các chỉ số này không nhằm so sánh policy, mà nhằm chứng minh framework đã chạy đúng.

Ở tầng đánh giá scheduling, các chỉ số có thể gồm total reward, average waiting time, average slowdown, utilization và throughput, tùy theo những gì environment đã hỗ trợ. Trong báo cáo, không cần cố thu quá nhiều metric nếu chưa thật sự ổn định; tốt hơn là chọn ít chỉ số nhưng có thể giải thích được và có log hỗ trợ. Theo slide deck, episode metrics và scheduler comparison tables là những artifact được kỳ vọng phải có.

Ở tầng reproducibility, chỉ số phù hợp là mức độ nhất quán của các metric cuối episode khi chạy lặp lại cùng seed và config. Nếu có thể, có thể báo cáo thêm mức độ trùng khớp của action sequence hoặc số decision points giữa các lần chạy. Điều này sẽ làm phần đánh giá có chiều sâu hơn.

---

### 7. Artifact cần thu thập trong quá trình thử nghiệm

Mỗi thử nghiệm phải để lại một tập artifact rõ ràng. Điều này không chỉ phục vụ viết báo cáo mà còn là bằng chứng kỹ thuật trước hội đồng.

Artifact bắt buộc nên gồm: file cấu hình thí nghiệm, seed log, episode metrics ở dạng CSV, log runtime của episode, và một bảng so sánh giữa các policy trong benchmark. Nếu hệ thống có hỗ trợ, nên thêm learning curves hoặc step-level decision log để làm bằng chứng phụ trợ. Theo tài liệu của đề tài, đây là những đầu ra kỳ vọng cốt lõi của hệ thống.

Điều quan trọng là artifact phải được tổ chức sao cho một người đọc ngoài cuộc có thể nhìn vào và hiểu được thí nghiệm nào đã chạy với cấu hình nào. Nếu artifact bị rời rạc hoặc không gắn được với config và seed, thì giá trị của reproducibility sẽ giảm mạnh.

---

### 8. Cách trình bày kết quả trong báo cáo

Kết quả thử nghiệm không nên trình bày theo kiểu liệt kê rời rạc. Thay vào đó, nên chia thành ba cụm rõ ràng.

Cụm thứ nhất là **kết quả kiểm chứng kỹ thuật**, gồm bằng chứng rằng environment đã chạy end-to-end, observation và action đã được xử lý đúng, reward đã được tính và episode đã kết thúc đúng. Phần này có thể dùng log, bảng mô tả một episode và ảnh chụp đầu ra terminal hoặc file log.

Cụm thứ hai là **kết quả benchmark tối thiểu**, gồm bảng so sánh heuristic với policy thứ hai trên cùng workload. Phần này nên có một bảng rõ ràng về các metric và một đoạn phân tích ngắn giải thích sự khác biệt.

Cụm thứ ba là **kết quả reproducibility và debug**, gồm bằng chứng rằng khi giữ nguyên seed và config thì kết quả không thay đổi đáng kể, cùng với một ví dụ decision point được truy vết bằng log. Phần này rất hữu ích vì nó cho thấy framework đã chạm tới chất của một hệ thống nghiên cứu chứ không chỉ là demo kỹ thuật.

---

### 9. Cách diễn giải kết quả trước hội đồng

Khi trình bày với hội đồng, sinh viên không nên tự đặt mình vào thế phải chứng minh một thuật toán RL mạnh. Thay vào đó, nên dẫn dắt theo logic sau.

Thứ nhất, hệ thống đã được hiện thực thành công như một Gym-compatible event-driven environment cho scheduling simulation, thể hiện qua khả năng chạy end-to-end một episode. Thứ hai, hệ thống đã có khả năng benchmark tối thiểu giữa hai policy trong cùng điều kiện, nên giá trị của framework đã được kiểm chứng ở mức sử dụng thực tế. Thứ ba, hệ thống hỗ trợ lưu config, seed, metric và log decision point, do đó có thể tái lập và phân tích thí nghiệm.

Cách diễn giải này vừa trung thực, vừa bám đúng tiêu chí của đề tài, đồng thời tránh bị kéo vào tranh luận kiểu “vì sao RL chưa hơn heuristic”. Theo định hướng hiện tại, điều đó không phải tiêu chí chính để phán xét thành công hay thất bại của framework.

---

### 10. Kết luận 

Phần đánh giá và thử nghiệm của đề tài được xây dựng nhằm kiểm chứng PyBatGym như một prototype framework cho event-driven RL scheduling environment. Trọng tâm đánh giá không nằm ở việc tối ưu một thuật toán RL, mà ở việc chứng minh rằng môi trường có thể vận hành end-to-end, có thể dùng để benchmark tối thiểu giữa các policy, và có thể tái lập thí nghiệm thông qua config, seed và artifact. Thông qua các nhóm thử nghiệm gồm kiểm tra loop kỹ thuật, kiểm tra semantics của observation-action-reward, benchmark tối thiểu, reproducibility và debug decision point, phần đánh giá có thể cung cấp bằng chứng tương đối đầy đủ cho tính đúng đắn và giá trị nghiên cứu của hệ thống. Đây là hướng đánh giá phù hợp nhất với scope cuối cùng của đề tài cũng như với các tiêu chí mà tài liệu hiện tại đã xác lập.
