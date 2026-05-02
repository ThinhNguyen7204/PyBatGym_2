# Giải đáp toàn diện các câu hỏi và thắc mắc bảo vệ đề tài PyBatGym

Tài liệu này tổng hợp và trả lời trực tiếp toàn bộ các câu hỏi checklist, các câu hỏi đánh giá nghiên cứu, và các luận điểm phản biện (thắc mắc) được đặt ra trong 3 tài liệu: Định hướng kỹ thuật, Đánh giá (Evaluation), và Thiết kế Kỹ thuật (Technical Design). Đây sẽ là cẩm nang giúp bạn tự tin trả lời trước hội đồng.

---

## PHẦN 1: TRẢ LỜI CHECKLIST CUỐI CÙNG TRƯỚC KHI RA HỘI ĐỒNG (10 CÂU HỎI)

**1. Tôi đã chốt một mô tả nhất quán về PyBatGym chưa?**
*   **Trả lời:** Đã chốt. PyBatGym được định nghĩa nhất quán xuyên suốt mã nguồn (README) và tài liệu là: *"Một framework môi trường học tăng cường (Gym-compatible environment) cho scheduling simulation theo hướng event-driven"*. Nó không phải là một bộ lập lịch, cũng không phải là công cụ thay thế BatSim, mà là một lớp chuẩn hóa giao tiếp (interface) giữa thuật toán AI (Policy) và hệ thống mô phỏng (Simulator).

**2. Tôi đã giải thích rõ vì sao PyBatGym không lấy BatSim làm core chưa?**
*   **Trả lời:** Đã giải thích rõ qua thiết kế **Architecture Adapter Pattern** (File `pybatgym/batsim_adapter.py` và `pybatgym/real_adapter.py`). BatSim chỉ đóng vai trò là "Backend đầu tiên" để kiểm chứng vòng lặp `reset-step`. Logic MDP (trạng thái, hành động, phần thưởng) nằm hoàn toàn ở `Environment Core` và độc lập với engine mô phỏng. Việc có cả `MockAdapter` và `RealBatsimAdapter` minh chứng rõ hệ thống linh hoạt và không bị khóa cứng vào lõi của BatSim.

**3. Tôi đã có sơ đồ kiến trúc khớp với code hiện tại chưa?**
*   **Trả lời:** Về mặt mô tả văn bản, tài liệu *Technical Design* đã nêu rất chuẩn xác 4 lớp kiến trúc (Simulator Backend, Integration, Environment Core, Experiment Support). Tuy nhiên, **bạn cần vẽ thêm một sơ đồ hình ảnh (Architecture Diagram)** đưa vào Slide bảo vệ để trực quan hóa văn bản này, đảm bảo hội đồng thấy ngay kiến trúc 4 lớp tương ứng với các module trong code.

**4. Tôi đã chốt observation, action, reward, episode chưa?**
*   **Trả lời:** Đã chốt chặt chẽ trong mã nguồn. 
    *   **Observation:** Định nghĩa duy nhất qua `DefaultObservationBuilder` (`observation.py`), gồm feature toàn cục (thời gian, tài nguyên) và thuộc tính của Top-K jobs.
    *   **Action:** Chuẩn hóa thành `Discrete(K+1)` thông qua `DefaultActionMapper` (`action.py`), tức là chọn 1 job trong Top-K để thực thi.
    *   **Reward:** Sử dụng một công thức cố định ưu tiên giảm Waiting Time và tăng Utilization thông qua `DefaultRewardCalculator` (`reward.py`).

**5. Tôi đã có code chạy end-to-end một episode chưa?**
*   **Trả lời:** Đã có. Các script trong thư mục `examples/` (`quickstart.py`, `test_real.py`, `train_ppo_trace.py`) đều có khả năng khởi tạo môi trường, nhận workload, chạy vòng lặp agent-simulator, và kết thúc episode (terminated=True) một cách hoàn chỉnh.

**6. Tôi đã có ít nhất một baseline heuristic chưa?**
*   **Trả lời:** Đã có tới 3 baseline heuristic. Module `pybatgym/plugins/benchmark.py` cung cấp sẵn FCFS (First-Come-First-Serve), SJF (Shortest-Job-First), và EASY Backfilling.

**7. Tôi đã có policy thứ hai để benchmark chưa?**
*   **Trả lời:** Đã có. Trong `examples/train_ppo_trace.py`, hệ thống tích hợp thành công PPO Agent từ Stable-Baselines3, tiến hành huấn luyện và so sánh kết quả trực tiếp với heuristic.

**8. Tôi đã có artifact gồm config, seed, metrics và comparison table chưa?**
*   **Trả lời:** Đã hoàn thiện toàn bộ luồng xuất Artifact:
    *   **Config:** Quản lý tập trung qua Pydantic (`pybatgym.config`).
    *   **Seed:** Môi trường hoàn toàn tái lập được thông qua việc truyền tham số `seed` vào `env.reset()`.
    *   **Metrics & Comparison:** Metric lưu theo step qua `CSVLoggerPlugin`, biểu đồ qua `TensorBoardLoggerPlugin`, và kết quả in ra bảng so sánh (Comparison Table) ngay tại Terminal ở cuối file test.

**9. Tôi đã có một ví dụ log decision point để giải thích hành vi policy chưa?**
*   **Trả lời:** Hiện tại code đã có `CSVLoggerPlugin` và log màu ANSI trên Terminal in ra trạng thái từng bước. Tuy nhiên, để trả lời xuất sắc câu hỏi này trước hội đồng, **bạn nên chuẩn bị sẵn một slide chụp màn hình** minh họa một "decision point" cụ thể (VD: Step 5 có 3 job chờ, tính năng của từng job là gì, agent đã chọn job nào, và reward nhận được là bao nhiêu).

**10. Tôi đã có cách giải thích giới hạn của đề tài mà không bị xem là né tránh chưa?**
*   **Trả lời:** Đã có. Bạn sẽ định vị đề tài này là một **"Vertical slice research prototype"**. Tức là: vì nguồn lực chỉ có 1 sinh viên, thay vì làm dàn trải nhiều backend nhưng lỗi, bạn tập trung đi sâu theo chiều dọc, chứng minh một pipeline hoàn chỉnh (từ event-driven loop -> đánh giá heuristic -> training RL -> xuất artifact) chạy thành công rực rỡ với 1 backend (BatSim) và 1 variant MDP cố định.

---

## PHẦN 2: TRẢ LỜI CÁC CÂU HỎI NGHIÊN CỨU & ĐÁNH GIÁ (Từ tài liệu Evaluation)

**1. PyBatGym có vận hành được như một Gym-compatible environment cho scheduling simulation hay không?**
*   **Trả lời:** Hoàn toàn được. Hệ thống tuân thủ nghiêm ngặt interface của Gymnasium (OpenAI Gym cũ): hỗ trợ `reset()`, `step()`, trả về đúng tuple `(observation, reward, terminated, truncated, info)`, cho phép bất kỳ thư viện RL chuẩn nào (như Stable-Baselines3) cũng có thể plug-and-play.

**2. PyBatGym có hỗ trợ được benchmark tối thiểu giữa các chính sách lập lịch khác nhau hay không?**
*   **Trả lời:** Có. Bằng chứng là script `train_ppo_trace.py` đã đặt FCFS, SJF, EASY Backfilling và PPO Agent vào chung một pipeline thử nghiệm (cùng workload, cùng platform, cùng hệ thống tính điểm) và trích xuất ra bảng so sánh công bằng về Avg Waiting Time, Slowdown và Utilization.

**3. PyBatGym có hỗ trợ reproducibility (tính tái lập) ở mức cơ bản hay không?**
*   **Trả lời:** Có. Mọi thành phần ngẫu nhiên (nếu có) đều bị kiểm soát bởi `seed`. File workload JSON, cấu hình Pydantic và random seed được lưu trữ và nạp lại chính xác, đảm bảo chạy 10 lần sẽ ra đúng 10 kết quả giống hệt nhau.

**4. PyBatGym có cung cấp đủ log để giải thích decision point hay không?**
*   **Trả lời:** Có. `CSVLoggerPlugin` ghi nhận chính xác trạng thái tại từng thời điểm quyết định (Decision Point), không phải theo chu kỳ thời gian (Time-step). Log ghi lại Action Agent chọn, số lượng pending jobs và Reward sinh ra sau đó.

---

## PHẦN 3: GIẢI ĐÁP CÁC THẮC MẮC CHUYÊN SÂU & PHẢN BIỆN (Từ tài liệu Technical Design)

**Thắc mắc 1: "PyBatGym có phải chỉ là một lớp wrapper gọi API của BatSim không?"**
*   **Giải thích bảo vệ:** Không. Một wrapper chỉ đơn giản là bọc hàm C++ thành hàm Python. PyBatGym làm nhiều hơn thế rất nhiều: Nó định nghĩa lại bài toán (formulate MDP). Nó quy định khi nào agent được hành động (Event-driven decision point). Nó có `Observation Encoder` nén dữ liệu khổng lồ của mô phỏng thành 1 vector số, `Action Processor` biên dịch số nguyên thành lệnh Dispatch, và `Reward Engine` tự tính điểm. Lớp lõi này không hề biết BatSim là gì, nó chỉ giao tiếp với khái niệm "Backend Adapter".

**Thắc mắc 2: "Môi trường Event-driven khác gì môi trường Time-step thông thường?"**
*   **Giải thích bảo vệ:** Trong các game Atari, `step()` tiến tới 1 khung hình (1/60s). Nhưng trong Scheduling, nếu 10 giây không có job nào đến, agent không cần quyết định gì cả. Môi trường Event-driven của PyBatGym giúp bỏ qua các khoảng thời gian trống. `step()` chỉ dừng lại ở các **Decision Point** có ý nghĩa (khi có Job mới đến, hoặc khi có tài nguyên được giải phóng). Điều này giúp policy học nhanh và sát với bài toán thực tế hơn.

**Thắc mắc 3: "Agent thực sự quan sát gì và điều khiển cái gì? Liệu nó có kiểm soát BatSim không?"**
*   **Giải thích bảo vệ:** Agent **không** kiểm soát BatSim ở tầng thấp. 
    *   **Quan sát:** Nó chỉ thấy một bản tóm tắt rút gọn (Observation Vector) về lượng tài nguyên trống và đặc trưng của `K` jobs đang chờ đầu tiên.
    *   **Điều khiển:** Agent chỉ việc chọn một con số `a` (từ 0 đến K). PyBatGym sẽ dịch `a` đó thành lệnh "Phân bổ Job thứ `a` vào máy chủ trống". Việc chạy mô phỏng vật lý tiếp theo do BatSim lo. Cơ chế này giúp giới hạn không gian hành động, tránh làm RL bị bối rối.

**Thắc mắc 4: "Làm sao biết công thức Reward là hợp lý, có phải đang ép số không?"**
*   **Giải thích bảo vệ:** Reward được tính độc lập trong mô-đun `Reward Engine`. Công thức hiện tại được thiết kế tuyến tính, thưởng khi tỷ lệ sử dụng máy chủ (Utilization) tăng và phạt nhẹ khi thời gian chờ (Waiting time) bị dồn ứ. Bằng chứng kiểm chứng nằm ở biểu đồ TensorBoard: Khi Reward tăng, ta thấy Utilization tăng và Avg Waiting Time thực tế giảm, chứng tỏ Reward đã phản ánh đúng (align) với mục tiêu tối ưu.

**Thắc mắc 5: "Nếu RL (PPO) chạy ra kết quả không tốt hơn Heuristic (như EASY Backfill), thì framework này có vô dụng không?"**
*   **Giải thích bảo vệ:** Hoàn toàn không. Mục tiêu của đồ án cuối kỳ là **chứng minh Framework (môi trường PyBatGym) hoạt động đúng**, chứ không phải chứng minh thuật toán PPO vô địch. Việc Framework báo cáo đúng rằng PPO đang ngang bằng hoặc thua EASY Backfilling chính là bằng chứng cho thấy Framework đã đo lường chính xác, minh bạch. Chuyện tạo ra một Policy RL vượt qua Heuristic tối ưu của HPC là bài toán thuộc cấp độ nghiên cứu dài hạn, Framework chỉ cung cấp "sân chơi chuẩn" để các nhà nghiên cứu thực hiện điều đó.
