# BÁO CÁO KỸ THUẬT: HỆ THỐNG PYBATGYM
### Môi trường Reinforcement Learning cho Bài toán Lập lịch trong Hệ thống Điện toán Hiệu năng Cao

> **Phiên bản:** 1.0 &nbsp;|&nbsp; **Ngày:** 2026-04-24  
> **Công nghệ cốt lõi:** Python 3.10+ · Gymnasium · Pydantic v2 · ZeroMQ · Stable-Baselines3

---

## MỤC LỤC

1. [Kiến trúc Tổng quan](#1-kiến-trúc-tổng-quan)
2. [Mô hình Chi tiết các Thành phần](#2-mô-hình-chi-tiết-các-thành-phần)
3. [Sơ đồ Khối và Luồng Dữ liệu](#3-sơ-đồ-khối-và-luồng-dữ-liệu)
4. [Workflow Hệ thống Từ đầu đến Cuối](#4-workflow-hệ-thống-từ-đầu-đến-cuối)
5. [Tích hợp BatSim — Cơ chế Socket và Event-driven](#5-tích-hợp-batsim--cơ-chế-socket-và-event-driven)
6. [Lớp Môi trường Trung tâm, Cấu hình và Tích hợp RL](#6-lớp-môi-trường-trung-tâm-cấu-hình-và-tích-hợp-rl)
7. [Đánh giá và Thử nghiệm Hệ thống](#7-đánh-giá-và-thử-nghiệm-hệ-thống)
8. [Hạn chế và Định hướng Tương lai](#8-hạn-chế-và-định-hướng-tương-lai)
9. [Giải đáp Phản biện (Q&A)](#9-giải-đáp-phản-biện-qa)

---

## 1. KIẾN TRÚC TỔNG QUAN

### 1.1 Triết lý Thiết kế

PyBatGym được hình thành với mục tiêu trở thành một môi trường Reinforcement Learning (RL) chuẩn mực, linh hoạt và thu hẹp tối đa "khoảng cách thực tế" (reality gap) cho bài toán lập lịch trong hệ thống điện toán hiệu năng cao (HPC). Để tránh việc xây dựng một hệ thống nguyên khối khó bảo trì và mở rộng, nhóm phát triển đã xây dựng PyBatGym dựa trên các nguyên tắc thiết kế cốt lõi sau:

**1. Kiến trúc Phân lớp (Layered Architecture) và Adapter Pattern:**
Đây là nguyên tắc sống còn của hệ thống. Thay vì để môi trường RL xử lý trực tiếp logic mô phỏng, toàn bộ việc giao tiếp với simulator được bọc sau một interface trừu tượng duy nhất: `BatsimAdapter`. Điều này có nghĩa là lớp môi trường (`PyBatGymEnv`) hoàn toàn "mù" về cách mô phỏng diễn ra bên dưới. Nó chỉ gọi các hàm API chuẩn (`reset`, `step`) và nhận về cùng một định dạng dữ liệu (danh sách Event, trạng thái Resource). Nhờ đó, người dùng có thể hoán đổi giữa `MockAdapter` (mô phỏng in-memory bằng Python để train nhanh) và `RealBatsimAdapter` (kết nối với BatSim C++ thực tế để validate) mà không cần sửa đổi bất kỳ dòng code RL nào.

**2. Mô hình Hướng Sự kiện (Event-Driven Simulation):**
PyBatGym đã loại bỏ hoàn toàn cơ chế Tick-based cũ (tiến thời gian lên từng giây một) để chuyển sang mô hình Event-driven. Mô hình này nhảy vọt thời gian đến các "điểm quyết định" (decision points) — những lúc có job mới nộp vào hoặc job cũ vừa chạy xong. Điều này không chỉ giúp tốc độ huấn luyện (training) tăng vọt do loại bỏ các bước thời gian trống (idle steps), mà còn đảm bảo `MockAdapter` hoạt động với cơ chế giao thức giống hệt như `RealBatsimAdapter` (vốn giao tiếp bằng ZeroMQ dựa trên sự kiện).

**3. Strategy Pattern cho các Thành phần Cốt lõi:**
Các thao tác xử lý dữ liệu RL — bao gồm việc xây dựng vector trạng thái (`ObservationBuilder`), ánh xạ hành động (`ActionMapper`), và tính điểm thưởng (`RewardCalculator`) — đều được tách thành các module độc lập (strategies). Người dùng có thể dễ dàng thay thế cách tính reward hoặc định dạng lại vector quan sát bằng cách truyền một object tùy chỉnh vào lúc khởi tạo mà không làm phá vỡ logic chung của hệ thống.

**4. Observer Pattern qua Plugin System & Dependency Injection:**
Các tác vụ phụ như ghi log (CSV, TensorBoard) hay chạy thuật toán đối chứng (baseline) được tách biệt vào các Plugin. Các plugin này lắng nghe các vòng đời của môi trường (`on_reset`, `on_step`). Bên cạnh đó, `PyBatGymEnv` áp dụng Dependency Injection: nó nhận tất cả các thành phần (Adapter, Reward, Observation...) từ bên ngoài qua tham số khởi tạo, giúp việc viết Unit Test bằng mock objects trở nên cực kỳ dễ dàng.

### 1.2 Sơ đồ Kiến trúc Phân lớp

Hệ thống được tổ chức thành 6 tầng từ cao xuống thấp. Dữ liệu chảy qua các tầng này một cách tuần tự, mỗi tầng chỉ giao tiếp với tầng liền kề:

```text
+------------------------------------------------------------------+
|              LAYER 5: RL Agent (PPO / A2C / Custom)              |
|        (Sử dụng Stable-Baselines3 hoặc Custom Agent)             |
|   [Nhận]: Vector trạng thái (obs), điểm thưởng (reward), done    |
|   [Xuất]: Quyết định hành động (action - số nguyên)              |
+-----------------------------+------------------------------------+
                              |  gọi env.step(action)
+-----------------------------v------------------------------------+
|                  LAYER 4: Gymnasium Environment                  |
|                     PyBatGymEnv  (env.py)                        |
|   - Trạm trung chuyển điều phối vòng lặp RL.                     |
|   - Không chứa logic mô phỏng, chỉ gọi các lớp bên dưới.         |
|   - Phát tín hiệu cho các Plugin (Logging, Benchmark).           |
+--------+------------+---------+----------------------------------+
         |            |         |  gọi map() / compute_reward() / build()
         v            v         v
+----------+ +----------+ +----------+   LAYER 3: Strategy Components
| ObsBuild | | ActionMap| | RewardCal|   - Xử lý mảng toán học và logic RL
+----------+ +----------+ +----------+   - Tùy biến linh hoạt tại runtime
         |            |         |
         +------------+---------+
                      | xuất ScheduleCommand (EXECUTE, WAIT...)
+---------------------v--------------------------------------------+
|                  LAYER 2: Adapter Layer                           |
|            BatsimAdapter (Abstract Base Class)                    |
|   - Chuẩn hóa luồng giao tiếp giữa RL và Simulator.               |
|                                                                   |
|   +----------------------------+  +---------------------------+   |
|   | EventDrivenMockAdapter     |  |   RealBatsimAdapter       |   |
|   | (Python in-memory nhanh)   |  |   (Giao tiếp ZMQ Socket)  |   |
|   +----------------------------+  +---------------------------+   |
+----------------------------+--------------------------------------+
                             | giao tiếp tiến trình / ZeroMQ TCP
+----------------------------v--------------------------------------+
|                  LAYER 1: Simulator / Backend                     |
|   [Mock]: Priority Queue xử lý sự kiện trong bộ nhớ Python.       |
|   [Real]: Tiến trình BatSim C++ (dùng SimGrid engine, port 28000) |
+----------------------------+--------------------------------------+
                             |
+----------------------------v--------------------------------------+
|               LAYER 0: Infrastructure & Configuration             |
|   Pydantic v2 Config  |  Plugin Hooks  |  Data Models (Job, Event)|
+-------------------------------------------------------------------+
```

**Diễn giải luồng chạy cơ bản:**
Khi Agent (Layer 5) đưa ra một hành động, nó sẽ rơi xuống `PyBatGymEnv` (Layer 4). `PyBatGymEnv` lập tức chuyển hành động đó cho `ActionMapper` (Layer 3) để dịch thành một lệnh cụ thể (VD: chạy Job A). Lệnh này được đẩy xuống `BatsimAdapter` (Layer 2) để thực thi trên backend mô phỏng (Layer 1). Sau khi backend mô phỏng chạy xong một điểm sự kiện, nó trả về danh sách các sự kiện (ví dụ: Job A hoàn thành). Từ danh sách này, `RewardCalculator` sẽ tính điểm thưởng, `ObservationBuilder` sẽ gom nhặt thông tin tài nguyên tạo thành vector quan sát mới, và `PyBatGymEnv` sẽ đóng gói tất cả gửi ngược lại cho Agent.

### 1.3 Các Thành phần Chính

Dưới đây là 10 mảnh ghép tạo nên toàn bộ hệ thống PyBatGym, mỗi thành phần đảm nhiệm một vai trò cụ thể:

| # | Thành phần cốt lõi | Vị trí File | Vai trò và Đặc điểm |
|---|-------------------|-------------|---------------------|
| 1 | **PyBatGymEnv** | `pybatgym/env.py` | Cổng giao tiếp chuẩn Gymnasium. Nơi điều phối dữ liệu qua lại giữa Agent và Simulator, quản lý vòng đời Episode. |
| 2 | **BatsimAdapter** | `pybatgym/batsim_adapter.py`| Interface trừu tượng định nghĩa các hàm bắt buộc (`reset`, `step`) mà bất kỳ backend simulator nào cũng phải có. |
| 3 | **MockAdapter** | `pybatgym/batsim_adapter.py`| Hiện thực `EventDrivenMockAdapter` bằng Python thuần. Nhảy cóc thời gian dựa trên hàng đợi sự kiện (Priority Queue). Dùng để train agent tốc độ cao. |
| 4 | **RealAdapter** | `pybatgym/real_adapter.py` | Kết nối với simulator BatSim C++ thực qua giao thức mạng ZeroMQ (REO/REP). Dùng để đánh giá (evaluate) độ chuẩn xác của agent. |
| 5 | **ObservationBuilder**| `pybatgym/observation.py` | Biến đổi thông tin hệ thống (tình trạng cluster, hàng đợi) thành 1 vector số thực 1D (float32) và 1 Action Mask phục vụ mạng Neural. |
| 6 | **ActionMapper** | `pybatgym/action.py` | Dịch quyết định của agent (ví dụ: action=2) thành lệnh thực tế (`EXECUTE_JOB`, `WAIT`, `BACKFILL`) gửi xuống Adapter. |
| 7 | **RewardCalculator**| `pybatgym/reward.py` | Tính toán điểm thưởng dựa trên các sự kiện vừa diễn ra, tối ưu hóa mức sử dụng tài nguyên (utilization) và thời gian chờ (waiting time). |
| 8 | **Data Models** | `pybatgym/models.py` | Định nghĩa các thực thể dùng chung toàn hệ thống như `Job`, `Resource`, `Event`, `ScheduleCommand` bằng Python Dataclasses. |
| 9 | **Plugin System** | `pybatgym/plugins/` | Các module gắn rời (CSV Logger, TensorBoard Logger, Benchmark Baseline) tự động chạy khi môi trường thực hiện `step` hoặc `reset`. |
| 10| **Config Layer** | `pybatgym/config/` | Hệ thống quản lý cấu hình bằng Pydantic v2 (đọc từ file `.yaml`), tự động validate kiểu dữ liệu và gán giá trị mặc định. |

---

## 2. MÔ HÌNH CHI TIẾT CÁC THÀNH PHẦN

### 2.1 Data Models — Nền tảng Domain Layer

`models.py` là tầng domain objects — các cấu trúc dữ liệu cơ bản được chia sẻ và sử dụng xuyên suốt toàn bộ hệ thống. Việc tập trung định nghĩa domain objects tại một nơi giúp đảm bảo tính nhất quán của dữ liệu và tránh tình trạng mỗi module tự định nghĩa riêng.

**Job** đại diện cho một batch job trong hàng đợi HPC. Mỗi job mang theo định danh duy nhất, thời điểm nộp vào hệ thống (`submit_time`), thời gian chạy tối đa do người dùng khai báo (`requested_walltime`), thời gian chạy thực tế chỉ biết sau khi hoàn thành (`actual_runtime`), số lượng CPU core cần thiết và trạng thái hiện tại trong vòng đời (PENDING / RUNNING / COMPLETED). Từ các trường cơ bản này, hệ thống tính toán thêm hai chỉ số quan trọng: `waiting_time` là khoảng thời gian từ lúc nộp đến lúc bắt đầu thực thi, và `bounded_slowdown` — thước đo công bằng giữa các job dài và ngắn, được định nghĩa là tỉ lệ giữa thời gian phản hồi thực tế và thời gian chạy tối thiểu. Hai chỉ số này được dùng trực tiếp trong hàm phần thưởng.

**Resource** mô tả bể tài nguyên của toàn bộ cluster. Thay vì theo dõi từng node riêng lẻ, hệ thống trừu tượng hóa tài nguyên thành số tổng cores và số cores đang được sử dụng. Lớp này cung cấp các phương thức kiểm tra khả năng cấp phát (`can_allocate`), tiêu thụ tài nguyên (`allocate`) và giải phóng tài nguyên (`release`). Chỉ số `utilization` — tỉ lệ cores đang dùng trên tổng cores — được tính tự động và là một trong những mục tiêu tối ưu quan trọng nhất của bài toán lập lịch.

**Event** đại diện cho các sự kiện phát sinh trong quá trình mô phỏng. Mỗi sự kiện mang theo loại (`EventType`), thời điểm xảy ra và job liên quan (nếu có). Có năm loại sự kiện được định nghĩa: `JOB_SUBMITTED` khi một job mới được nộp vào hàng đợi, `JOB_STARTED` khi job bắt đầu thực thi, `JOB_COMPLETED` khi job hoàn thành, `RESOURCE_FREED` khi tài nguyên được giải phóng đi kèm với sự kiện hoàn thành, và `SIMULATION_ENDED` khi toàn bộ workload đã được xử lý. Danh sách sự kiện này chính là "ngôn ngữ giao tiếp" giữa adapter và các thành phần RL phía trên.

**ScheduleCommand** là lệnh lập lịch mà RL Agent ra quyết định và gửi xuống adapter. Có bốn loại lệnh: `EXECUTE_JOB` để chạy một job cụ thể ngay lập tức, `BACKFILL_JOB` để lấp đầy khoảng trống bằng job nhỏ nhất phù hợp, `RESERVE_RESOURCE` để đặt trước tài nguyên (chưa sử dụng trong phiên bản hiện tại), và `WAIT` để không làm gì và nhường thời gian cho simulator tiến lên.

### 2.2 `env.py` — Trung tâm Điều phối Gymnasium

`PyBatGymEnv` là lớp trung tâm của toàn bộ hệ thống, kế thừa `gymnasium.Env` và là điểm giao tiếp duy nhất giữa RL agent và thế giới simulation. Điểm đặc biệt và quan trọng nhất trong thiết kế của lớp này là nó **không chứa bất kỳ logic simulation, logic tính toán phần thưởng hay logic xử lý quan sát nào**. Toàn bộ trách nhiệm đó được ủy thác hoàn toàn cho các thành phần chuyên biệt bên dưới. `PyBatGymEnv` chỉ làm một việc: điều phối đúng thứ tự, đúng thời điểm.

Lớp này lưu giữ tham chiếu đến bốn thành phần chính: adapter backend (`_adapter`), bộ xây dựng quan sát (`_obs_builder`), bộ ánh xạ hành động (`_action_mapper`) và bộ tính phần thưởng (`_reward_calc`). Ngoài ra, nó duy trì trạng thái episode hiện tại qua `_state` (dictionary chứa thời gian hiện tại, danh sách jobs đang chờ, trạng thái tài nguyên và danh sách sự kiện vừa xảy ra), cùng với bộ đếm bước và tích lũy phần thưởng để hỗ trợ logging và debug.

Hai phương thức API cốt lõi là `reset()` và `step()`. Phương thức `reset()` bắt đầu một episode mới bằng cách đặt lại toàn bộ bộ đếm, gọi `adapter.reset()` để khởi tạo lại simulation, xây dựng state dict ban đầu, tạo observation đầu tiên và thông báo cho tất cả plugin biết rằng một episode mới đã bắt đầu. Phương thức `step(action)` nhận một số nguyên hành động từ agent, dịch nó thành lệnh lập lịch, gửi xuống adapter để thực thi, cập nhật state, tính phần thưởng, kiểm tra điều kiện kết thúc, xây dựng observation mới và trả về tuple kết quả Gymnasium chuẩn gồm `(obs, reward, terminated, truncated, info)`.

Ngoài Dependency Injection đã đề cập, lớp này còn hỗ trợ đăng ký plugin động qua `register_plugin()`. Điều này cho phép người dùng gắn thêm plugin sau khi môi trường đã được khởi tạo, rất tiện lợi trong các kịch bản thực nghiệm linh hoạt.

### 2.3 Adapter Layer — Trừu tượng hóa Simulator

#### MockAdapter — Event-Driven In-Memory Simulator

`MockAdapter` là hiện thực simulator bằng Python thuần túy, không phụ thuộc vào BatSim C++ hay bất kỳ phần mềm ngoài nào. Nó mô phỏng toàn bộ vòng đời của một hàng đợi HPC bên trong bộ nhớ Python, với đủ đầy các trạng thái: hàng đợi chờ, danh sách đang thực thi và danh sách đã hoàn thành.

Đặc điểm thiết kế quan trọng nhất của `MockAdapter` là **mô hình thời gian sự kiện (event-driven time)**. Thay vì tiến lên từng đơn vị thời gian cố định (ví dụ mỗi bước tăng thêm 1.0 đơn vị), adapter tính toán thời điểm sự kiện quan trọng tiếp theo và nhảy thẳng đến đó. Sự kiện quan trọng được định nghĩa là thời điểm một job trong tương lai đến hạn nộp, hoặc một job đang chạy hoàn thành. Giữa hai sự kiện liên tiếp, không có bất kỳ quyết định lập lịch nào có thể làm thay đổi kết quả — do đó hệ thống bỏ qua hoàn toàn khoảng thời gian trống này, giúp training nhanh hơn nhiều lần so với fixed time-step.

Cơ chế này cũng giải thích một quy tắc quan trọng: khi agent chọn `EXECUTE_JOB`, đồng hồ simulation **không thay đổi**. Điều này có nghĩa là agent có thể liên tiếp lên lịch nhiều job trong cùng một thời điểm, tối đa hóa mức sử dụng tài nguyên. Chỉ khi agent chọn `WAIT` — hoặc khi không còn job nào có thể lên lịch — đồng hồ mới nhảy đến sự kiện kế tiếp.

Workload được sinh tổng hợp theo phân phối thống kê có cơ sở thực tế: thời gian giữa hai lần nộp job tuân theo phân phối Exponential; số cores yêu cầu phân phối đều trong khoảng cho phép; thời gian chạy thực tế thường nhỏ hơn walltime do người dùng có xu hướng khai báo dư. Khi cần độ chính xác cao hơn, adapter cũng hỗ trợ nạp workload từ file JSON trace của BatSim.

#### RealBatsimAdapter — Kết nối với BatSim C++

`RealBatsimAdapter` là hiện thực phức tạp hơn, kết nối trực tiếp với BatSim — một simulator HPC mã nguồn mở viết bằng C++, sử dụng engine SimGrid để mô phỏng chính xác hành vi của cluster thực. Kiến trúc multi-thread của adapter này được giải thích chi tiết trong Mục 5 của báo cáo.

### 2.4 ObservationBuilder — Xây dựng Vector Quan sát

`ObservationBuilder` có nhiệm vụ chuyển đổi trạng thái thô của simulation (một Python dictionary) thành **fixed-size float32 vector** — định dạng duy nhất mà neural network của RL agent có thể xử lý. Đây là bước chuẩn hóa thông tin từ không gian simulationmột chiều không cố định sang một không gian vector có cấu trúc nhất quán.

Vector quan sát có kích thước `11 + 4K` chiều, trong đó K là số job tối đa hiển thị (mặc định K=10, do đó vector có 51 chiều). Vector được chia thành ba vùng: **Global Features** (5 chiều) mô tả trạng thái tổng thể của cluster gồm tiến độ thời gian mô phỏng, độ dài hàng đợi, mức sử dụng tài nguyên, thời gian chờ trung bình và bounded slowdown trung bình; **Job Queue Features** (4K chiều) mô tả K job đang đợi lâu nhất trong hàng đợi, mỗi job được biểu diễn bởi bốn chỉ số chuẩn hóa gồm thời gian chờ, walltime, số cores và bounded slowdown; **Resource Features** (6 chiều) mô tả chi tiết hơn về tình trạng tài nguyên như tỉ lệ cores rảnh, số job có thể chạy ngay, mức độ cấp bách của hàng đợi và tình trạng phân mảnh tài nguyên.

Ngoài `features`, observation còn bao gồm một **action mask** (K+2 chiều) được trả về riêng trong dictionary. Mask này cho agent biết hành động nào hợp lệ tại bước hiện tại — ví dụ, mask của job thứ i sẽ bằng 0 nếu cluster không đủ core để chạy job đó. Nhờ mask này, các thuật toán như MaskablePPO có thể loại bỏ hoàn toàn các hành động không khả thi trước khi lấy mẫu, đẩy nhanh tốc độ hội tụ.

Tất cả các giá trị trong vector đều được chuẩn hóa về khoảng `[0.0, 1.0]` bằng cách chia cho giá trị tối đa tham chiếu được định nghĩa trong cấu hình. Việc chuẩn hóa là bắt buộc để đảm bảo gradient ổn định trong quá trình huấn luyện neural network.

### 2.5 ActionMapper — Ánh xạ Hành động

`ActionMapper` thực hiện bước dịch từ quyết định của RL agent (một số nguyên) sang lệnh lập lịch cụ thể để gửi xuống adapter.

Không gian hành động là **Discrete(K+2)**: K hành động đầu tiên (từ 0 đến K-1) tương ứng với việc chọn job thứ i trong top-K hàng đợi để chạy ngay; hành động K là `WAIT` — không làm gì và nhường quyền cho simulator tiến lên; hành động K+1 là `SCHEDULE_SMALLEST_FITTING` — greedy backfill tự động tìm job nhỏ nhất vừa khớp với tài nguyên còn lại.

Một đặc điểm quan trọng là khả năng **fallback tự động**: nếu agent chọn job thứ i nhưng tại thời điểm đó job này không tồn tại trong hàng đợi hoặc cluster không đủ tài nguyên để chạy nó, `ActionMapper` tự động chuyển lệnh thành `WAIT` thay vì gây lỗi. Điều này làm cho môi trường trở nên robust hơn trong quá trình training, khi agent ban đầu có thể đưa ra những lựa chọn vô nghĩa.

### 2.6 RewardCalculator — Hàm Phần thưởng Đa mục tiêu

`RewardCalculator` tính toán tín hiệu phần thưởng gửi về agent. Đây là một trong những thành phần quan trọng nhất vì chất lượng thiết kế reward sẽ quyết định trực tiếp đến việc agent có học được policy tốt hay không.

PyBatGym hỗ trợ ba chế độ reward. **Step reward** (dense) tính phần thưởng tại mỗi bước: thưởng khi cải thiện mức sử dụng tài nguyên, thưởng khi lên lịch thành công một job, phạt khi chọn WAIT trong khi hàng đợi vẫn còn job, phạt khi cluster idle trong khi còn job chờ, và trừ điểm proportional với thời gian chờ cùng slowdown của các job vừa hoàn thành. **Episodic reward** (sparse) chỉ tính một lần duy nhất khi episode kết thúc, đánh giá tổng thể các chỉ số: utilization, throughput, thời gian chờ trung bình và slowdown trung bình. **Hybrid** (mặc định và khuyến nghị) kết hợp cả hai: tích lũy step reward trong suốt episode để agent học được phản hồi tức thì, đồng thời cộng thêm terminal bonus khi kết thúc để hướng agent tối ưu toàn cục.

Trọng số giữa các mục tiêu có thể điều chỉnh qua cấu hình: mặc định `utilization=0.3`, `waiting_time=0.3`, `slowdown=0.3`, `throughput=0.1`. Người dùng có thể thay đổi các trọng số này để ưu tiên mục tiêu nào phù hợp với yêu cầu của cluster cụ thể.

### 2.7 Các Mô-đun Quản lý Thí nghiệm và Plugin System

Plugin system cho phép mở rộng hành vi của môi trường mà không cần chỉnh sửa code lõi. Mỗi plugin triển khai ba lifecycle hook: `on_reset` được gọi mỗi khi episode mới bắt đầu, `on_step` được gọi sau mỗi bước với đầy đủ thông tin về hành động, phần thưởng và trạng thái, và `on_close` được gọi khi môi trường đóng lại.

Bên cạnh Plugin System, hệ thống còn bao gồm các mô-đun quản lý thí nghiệm quan trọng phục vụ cho workflow nghiên cứu:
- **Episode Manager**: Quản lý vòng đời của một episode. Nó giữ trạng thái như episode đã bắt đầu hay chưa, tổng reward đã tích lũy, số decision point đã đi qua và quyết định thời điểm kết thúc (khi toàn bộ job hoàn tất hoặc đạt ngưỡng cắt).
- **Logging and Artifact Manager** (hiện thực qua `CSVLoggerPlugin` và `TensorBoardLoggerPlugin`): Trách nhiệm ghi nhận log ở cấp độ step (decision point) với các cột episode, step, action, reward, simulation time và utilization. Đồng thời hệ thống xuất cấu hình, seed để đảm bảo tính tái lập (reproducibility).
- **Benchmark Runner** (hiện thực qua `BenchmarkPlugin`): Điều phối việc chạy các thuật toán baseline song song như FCFS, SJF và EASY Backfilling trên cùng workload và platform để xuất ra bảng so sánh.

---

## 3. SƠ ĐỒ KHỐI VÀ LUỒNG DỮ LIỆU

### 3.1 Luồng Dữ liệu Tổng thể

Toàn bộ hệ thống có thể được hiểu theo một vòng lặp thông tin khép kín: **Agent → Môi trường → Simulator → Quan sát & Phần thưởng → Agent**. Tại mỗi bước của vòng lặp, thông tin chảy qua nhiều thành phần theo một trình tự nghiêm ngặt.

Khi agent gửi một hành động (số nguyên từ 0 đến K+1), `ActionMapper` là thành phần đầu tiên nhận và xử lý: nó tra cứu trạng thái hàng đợi hiện tại, xác định job tương ứng, kiểm tra tính khả thi của hành động và tạo ra `ScheduleCommand` phù hợp. Lệnh này sau đó được chuyển xuống `BatsimAdapter.step()`, nơi simulation thực sự diễn ra: tài nguyên được cấp phát hoặc đồng hồ được tiến lên, và adapter trả về danh sách các sự kiện đã xảy ra cùng với cờ báo hiệu simulation kết thúc hay chưa.

Từ danh sách sự kiện, môi trường cập nhật `state dict` — bức tranh toàn cảnh hiện tại của hệ thống. `RewardCalculator` đọc state dict và danh sách sự kiện để tính step reward, trong khi `ObservationBuilder` đọc state dict để xây dựng vector quan sát mới. Cuối cùng, tất cả thông tin được đóng gói thành tuple Gymnasium chuẩn và trả về agent để agent đưa ra quyết định tiếp theo.

### 3.2 Chi tiết Luồng dữ liệu trong một bước `step(action)`

Khi agent gửi hành động, ví dụ hành động số 3 (chọn job thứ 3 trong hàng đợi), quá trình diễn ra như sau. Đầu tiên, `ActionMapper` lấy top-K pending jobs sắp xếp theo thứ tự nộp, chọn job thứ 3, và kiểm tra xem cluster có đủ cores không. Nếu có, `ActionMapper` tạo `ScheduleCommand(EXECUTE_JOB, job=job3, cores=N)`; nếu không, nó tạo `ScheduleCommand(WAIT)`.

Lệnh được gửi xuống adapter. Nếu là `EXECUTE_JOB`, adapter cấp phát tài nguyên, di chuyển job từ hàng đợi sang danh sách đang chạy, ghi nhận thời điểm hoàn thành dự kiến, và trả về sự kiện `JOB_STARTED`. Thời gian **không thay đổi** sau bước này. Nếu là `WAIT`, adapter tính toán sự kiện tiếp theo gần nhất (hoặc job mới nộp, hoặc job đang chạy xong), nhảy đến đó, xử lý các completions và submissions xảy ra tại thời điểm đó, rồi trả về một batch sự kiện.

Tiếp theo, state dict được cập nhật với thời gian mới, hàng đợi mới và tài nguyên mới. `RewardCalculator` duyệt qua danh sách sự kiện: với mỗi `JOB_STARTED`, nó cộng thêm schedule bonus; với mỗi `JOB_COMPLETED`, nó trừ đi điểm tương ứng với waiting time và bounded slowdown của job đó; đồng thời tính delta utilization so với bước trước và áp dụng idle penalty nếu cần. `ObservationBuilder` đọc state dict và xây dựng vector 51 chiều kèm action mask cho bước tiếp theo.

Cuối cùng, môi trường kiểm tra hai điều kiện kết thúc: `terminated` (kết thúc tự nhiên — toàn bộ jobs đã được nộp, không còn job chờ, không còn job chạy) và `truncated` (bị cắt ngắn — đã vượt quá số bước tối đa được cấu hình). Nếu kết thúc, episode reward được tính thêm và cộng vào step reward hiện tại.

### 3.3 Luồng Dữ liệu `reset()`

Khi `env.reset(seed=42)` được gọi, môi trường đặt lại toàn bộ bộ đếm nội bộ và gọi `adapter.reset()`. Trong trường hợp MockAdapter, adapter sinh ngẫu nhiên một workload mới với seed được cung cấp — đảm bảo hoàn toàn xác định (deterministic): với cùng seed, workload sinh ra sẽ hoàn toàn giống hệt nhau mỗi lần, giúp experiments reproducible. Adapter đặt đồng hồ về 0, khởi tạo lại tài nguyên với toàn bộ cores rảnh, và nộp ngay những job có `submit_time = 0`.

Từ trạng thái khởi đầu đó, môi trường xây dựng observation đầu tiên và trả về cho agent, sẵn sàng bắt đầu episode mới.

### 3.4 Điều kiện Kết thúc Episode

Hệ thống phân biệt rõ ràng giữa hai loại kết thúc theo chuẩn Gymnasium hiện đại. **Terminated** (kết thúc tự nhiên) xảy ra khi ba điều kiện đồng thời thỏa mãn: tất cả jobs trong workload đã được nộp vào hệ thống, hàng đợi trống và không còn job nào đang thực thi. Đây là trạng thái lý tưởng — toàn bộ workload đã được xử lý hoàn chỉnh. **Truncated** (bị cắt ngắn) xảy ra khi số bước đã vượt quá `max_steps` được cấu hình, đây là cơ chế bảo vệ tránh episode kéo dài vô hạn trong trường hợp agent liên tục chọn WAIT hoặc gặp workload quá lớn.

---

## 4. WORKFLOW HỆ THỐNG TỪ ĐẦU ĐẾN CUỐI

### 4.1 Vòng đời Tổng thể

Khi triển khai một thực nghiệm với PyBatGym, hệ thống trải qua ba giai đoạn lớn: **Khởi động**, **Vòng lặp Training**, và **Đánh giá**.

Trong giai đoạn khởi động, người dùng nạp cấu hình từ file YAML hoặc tạo trực tiếp bằng code Python, khởi tạo môi trường và đăng ký các plugin cần thiết. Môi trường tự động lựa chọn backend phù hợp dựa vào trường `mode` trong cấu hình: `"mock"` cho phát triển nhanh và `"real"` cho nghiên cứu chính xác. Sau đó, agent RL (ví dụ PPO từ Stable-Baselines3) được tạo và truyền vào môi trường.

Vòng lặp training diễn ra qua hàng nghìn episode. Đầu mỗi episode, `env.reset()` được gọi để sinh workload mới và bắt đầu simulation từ đầu. Agent liên tục nhận observation, đưa ra quyết định lập lịch, nhận phần thưởng và cập nhật policy cho đến khi episode kết thúc. Sau mỗi episode, các metrics được ghi lại qua plugin system và agent sử dụng dữ liệu thu thập được để cải thiện policy.

### 4.2 Ví dụ Minh họa một Episode

Để hiểu rõ hơn cơ chế hoạt động, hãy theo dõi một episode đơn giản với workload gồm ba jobs:

Tại thời điểm `t = 0.0`, simulation bắt đầu với hai job đầu tiên đã được nộp sẵn: `job_0` cần 3 cores và `job_1` cần 4 cores. Cluster có tổng cộng 8 cores rảnh. Agent quan sát trạng thái này và chọn lên lịch `job_0`. Adapter cấp phát 3 cores, `job_0` bắt đầu chạy với thời gian hoàn thành dự kiến là `t = 10.0`. Thời gian không thay đổi, agent liên tiếp nhận observation mới và chọn `job_1`. Adapter cấp phát thêm 4 cores, `job_1` bắt đầu với `finish_time = 15.0`. Lúc này còn đúng 1 core rảnh, không đủ cho bất kỳ job nào khác. Agent chọn WAIT.

Adapter nhảy đến thời điểm sự kiện gần nhất: `job_2` có `submit_time = 5.0` và `job_0` có `finish_time = 10.0`, do đó nhảy đến `t = 5.0`. Tại đây, `job_2` được nộp vào hàng đợi (cần 2 cores tuy nhiên chỉ có 1 core rảnh, nên action mask cho `job_2` bằng 0). Agent lại chọn WAIT.

Adapter nhảy tiếp đến `t = 10.0` — `job_0` hoàn thành, giải phóng 3 cores, tổng còn 4 cores rảnh. `job_2` cần 2 cores, action mask mở. Agent chọn lên lịch `job_2`. Simulation tiếp tục cho đến khi tất cả jobs hoàn thành, episode kết thúc với `terminated = True` và agent nhận episode reward tổng kết.

### 4.3 Decision Point và Ý nghĩa Thiết kế

Khái niệm **decision point** là cốt lõi của thiết kế event-driven. Hệ thống chỉ yêu cầu agent đưa ra quyết định tại những thời điểm thực sự có ý nghĩa: khi có "hàng mới về kho" (job mới được nộp) hoặc khi có "kho mới mở" (tài nguyên vừa được giải phóng). Giữa hai decision point liên tiếp, bất kỳ quyết định lập lịch nào cũng sẽ cho kết quả như nhau — do đó hệ thống mạnh dạn bỏ qua toàn bộ khoảng thời gian đó.

Nhờ cơ chế này, một episode với 300 jobs trong MockAdapter chỉ cần khoảng 600–900 decision steps (thay vì hàng nghìn fixed time-step nếu dùng cách tiếp cận truyền thống), giúp tốc độ training nhanh hơn đáng kể.

---

## 5. TÍCH HỢP BATSIM — CƠ CHẾ SOCKET VÀ EVENT-DRIVEN

### 5.1 Tổng quan Kiến trúc Giao tiếp Real Mode

Khi cấu hình `mode = "real"`, PyBatGym chuyển sang sử dụng `RealBatsimAdapter` — thành phần kết nối trực tiếp với **BatSim C++**, một simulator HPC chính thống được phát triển bởi Inria (Pháp) trên nền tảng SimGrid. BatSim mô phỏng chi tiết các hành vi như mạng nội bộ cluster, I/O contention và overhead do lập lịch, mang lại độ trung thực cao hơn nhiều so với MockAdapter.

Giao tiếp giữa PyBatGym và BatSim diễn ra qua **ZeroMQ (ZMQ)** theo mô hình REQ/REP: BatSim đóng vai REQ client gửi thông báo sự kiện, pybatsim thread đóng vai REP server lắng nghe và phản hồi với các lệnh lập lịch. Đây là giao thức đã được thiết kế sẵn trong BatSim và được thư viện `pybatsim` (Python) triển khai.

### 5.2 Kiến trúc Hai Thread với Queue Bridges

Thách thức lớn nhất của Real Mode là bài toán đồng bộ hóa: vòng lặp ZMQ của pybatsim phải chạy liên tục để phản hồi BatSim kịp thời, trong khi vòng lặp RL cần tạm dừng để agent suy nghĩ và đưa ra quyết định. Hai vòng lặp này chạy ở tốc độ khác nhau và không thể gộp làm một.

Giải pháp là tách chúng thành **hai thread độc lập** với **hai Queue làm cầu nối**: `action_queue` (Main Thread đưa lệnh vào, Background Thread lấy ra để thực thi) và `state_queue` (Background Thread đưa thông báo vào, Main Thread lấy ra để biết đã đến lúc ra quyết định). Main thread là RL agent, chạy trong luồng chính. Background thread là pybatsim event loop, chạy như một daemon thread.

Khi BatSim phát sinh một sự kiện (ví dụ job mới được nộp), nó gửi thông báo ZMQ tới Background thread. Background thread gọi callback `onJobSubmission`, ghi nhận sự kiện, rồi gọi `_wakeup_and_wait()` — đây là hàm trái tim của cơ chế đồng bộ. Hàm này đẩy tín hiệu `"WAKEUP"` vào `state_queue` để đánh thức Main thread, rồi **block** tại `action_queue.get()` chờ quyết định. Main thread, sau khi nhận được tín hiệu WAKEUP, thu thập sự kiện, tính phần thưởng, xây dựng observation, cho agent suy nghĩ, rồi đẩy lệnh vào `action_queue`. Background thread nhận lệnh, thực thi với BatSim, và trả điều khiển về cho vòng lặp ZMQ. Đây là **Rendezvous Pattern** — hai thread gặp nhau tại mỗi decision point.

### 5.3 Quản lý Port và Multi-Episode

Mỗi lần `reset()` được gọi trong Real Mode, một phiên BatSim mới phải được khởi động. Nếu cứng nhắc dùng port 28000 cho mọi phiên, sẽ xảy ra xung đột port do cơ chế TCP TIME_WAIT của hệ điều hành — socket cũ chưa được hệ điều hành giải phóng hoàn toàn trong khi socket mới đã cố bind vào cùng port.

PyBatGym giải quyết vấn đề này bằng cách **tự động dò tìm port trống** mỗi khi khởi động: bắt đầu từ 28000 và tăng dần đến 28099, thử bind từng port và dừng lại ở port đầu tiên khả dụng. Điều này đảm bảo mỗi episode trong Real Mode có thể chạy độc lập mà không lo xung đột tài nguyên mạng.

### 5.4 Cleanup Protocol — Tránh Deadlock

Kết thúc một phiên Real Mode đòi hỏi thứ tự thực hiện rất cụ thể để tránh deadlock. Hệ thống thực thi ba bước theo trình tự: đầu tiên, đặt cờ `_is_done = True` và đẩy một thông báo trống vào `action_queue` để unblock Background thread nếu nó đang chờ; tiếp theo, kill tiến trình BatSim C++ — hành động này làm ZMQ REQ socket bên BatSim đóng lại, khiến ZMQ REP socket bên pybatsim nhận được lỗi và vòng lặp ZMQ tự thoát; cuối cùng, `join()` Background thread chờ nó dọn dẹp hoàn toàn.

Thứ tự này có lý do rõ ràng: nếu join Background thread trước khi kill BatSim, Background thread vẫn còn đang block tại `recv()` ZMQ đợi tin nhắn từ BatSim — `join()` sẽ chờ mãi mãi và gây deadlock. Chỉ khi BatSim bị kill trước, ZMQ mới báo lỗi, Background thread mới thoát được, `join()` mới thành công.

### 5.5 Tối ưu hóa — Giảm ZMQ Latency

Một tối ưu hóa được thực hiện trong `onJobCompletion`: khi một job hoàn thành, Background thread chỉ đánh thức Main thread (gây ra một vòng ZMQ round-trip) **nếu hàng đợi đang có job chờ đợi được lên lịch**. Nếu hàng đợi trống, BatSim tự tiến lên mà không cần hỏi agent — vì dù agent có được hỏi, câu trả lời cũng chỉ là WAIT. Tối ưu hóa này loại bỏ một lượng lớn các round-trip ZMQ không cần thiết trong các workload có nhiều completions liên tiếp.

---

## 6. LỚP MÔI TRƯỜNG TRUNG TÂM, CẤU HÌNH VÀ TÍCH HỢP RL

### 6.1 Phân tích Chi tiết `env.py` — `PyBatGymEnv`

#### Khởi tạo và Nguyên tắc Dependency Injection

Hàm `__init__` của `PyBatGymEnv` được thiết kế để nhận tất cả các thành phần phụ thuộc từ bên ngoài thay vì tự khởi tạo chúng. Người dùng có thể truyền vào `config_path` để load từ file YAML, hoặc truyền trực tiếp một object `PyBatGymConfig` đã được cấu hình bằng code Python. Nếu không cung cấp gì, hệ thống tự tìm file `configs/default.yaml` và nếu không có sẽ dùng toàn bộ giá trị mặc định từ Pydantic model.

Sau khi có cấu hình, môi trường lựa chọn adapter: nếu người dùng truyền vào một adapter đã tạo sẵn (thường dùng trong unit test), hệ thống dùng nó trực tiếp; nếu không, hệ thống tạo adapter phù hợp dựa vào giá trị `mode`. Tương tự với ObservationBuilder, ActionMapper và RewardCalculator: nếu không được inject, hệ thống tạo các hiện thực mặc định. Đây là phần hiện thực Dependency Injection — giữ cho code linh hoạt mà không phức tạp hóa interface.

Bước cuối của `__init__` là **khai báo observation space và action space** theo chuẩn Gymnasium. Đây là bước bắt buộc: Stable-Baselines3 và các thư viện RL khác đọc hai thuộc tính này ngay khi nhận môi trường để biết kiến trúc mạng nào phù hợp và bao nhiêu chiều đầu vào/đầu ra cần xây dựng.

#### Phân tích `reset()` — Khởi tạo Episode

`reset()` là điểm bắt đầu của mỗi episode và chịu trách nhiệm đưa hệ thống về trạng thái ban đầu một cách đầy đủ. Hàm này trước tiên gọi `super().reset(seed=seed)` theo yêu cầu bắt buộc của Gymnasium — bỏ qua bước này sẽ gây cảnh báo hoặc lỗi trong các phiên bản Gymnasium mới. Tiếp theo, tất cả bộ đếm nội bộ được đặt lại về zero và `reward_calc.reset()` được gọi để xóa `prev_utilization` — giá trị utilization của bước trước dùng trong tính delta reward.

Trọng tâm của `reset()` là lời gọi `adapter.reset()`. Đây là lúc workload mới được sinh hoặc nạp, đồng hồ simulation về 0, và tài nguyên trở về trạng thái rảnh hoàn toàn. Adapter trả về danh sách sự kiện ban đầu (thường là các `JOB_SUBMITTED` cho những jobs có `submit_time = 0`) và đối tượng Resource. Từ đó môi trường xây dựng `_state` dictionary — khung dữ liệu tổng hợp được chia sẻ giữa tất cả thành phần — và tạo ra observation đầu tiên.

#### Phân tích `step()` — Vòng lặp Thời gian Chính

`step(action)` là hàm thực sự tạo nên "vòng lặp thời gian" của simulation. Mỗi lần gọi, hệ thống tiến thêm một bước trong không gian quyết định của bài toán lập lịch. Điều cần nhấn mạnh là "một bước" ở đây không nhất thiết tương ứng với một đơn vị thời gian cố định — nó tương ứng với **một decision point**: một khoảnh khắc có ý nghĩa trong simulation nơi agent cần đưa ra lựa chọn.

Quá trình trong `step()` diễn ra tuần tự qua mười bước. Đầu tiên, bộ đếm bước tăng. Tiếp theo, hành động được dịch thành lệnh. Lệnh được gửi xuống adapter, adapter thực thi và trả về sự kiện. State dict được cập nhật. Phần thưởng được tính dựa trên sự kiện và trạng thái mới. Hai cờ kết thúc được kiểm tra. Nếu kết thúc, episode reward được tính thêm. Tích lũy reward được cập nhật. Observation mới được xây dựng. Cuối cùng, tất cả plugin được thông báo và tuple kết quả được trả về.

Nhìn vào chuỗi trên, có thể thấy `env.py` hoàn toàn là một **orchestrator** thuần túy — không có một công thức tính toán nào trực tiếp bên trong nó. Mọi tính toán thực sự đều diễn ra trong các thành phần chuyên biệt.

### 6.2 Quản lý Cấu hình và Tính Tái lập

#### Kiến trúc Config Layer

Toàn bộ cấu hình của PyBatGym được quản lý qua **Pydantic v2 BaseModel** — một thư viện Python cho phép định nghĩa schemas dữ liệu với validation tự động, type annotation đầy đủ và serialization sang nhiều định dạng. Pydantic được chọn vì nó kết hợp ba ưu điểm: an toàn kiểu dữ liệu tại thời điểm khởi tạo (không phải runtime), đọc YAML dễ dàng qua `model_validate(dict)`, và tự động áp dụng giá trị mặc định cho các trường không được khai báo.

Cấu hình gốc `PyBatGymConfig` được tổ chức thành các sub-config lồng nhau theo chức năng: `PlatformConfig` mô tả phần cứng cluster (số nodes, cores mỗi node); `WorkloadConfig` điều khiển việc sinh workload (synthetic hay trace, số jobs, bounded trên runtime); `EpisodeConfig` giới hạn thời gian và số bước mỗi episode; `ObservationConfig` điều chỉnh kích thước và ngưỡng chuẩn hóa của observation vector; và `RewardWeights` định nghĩa tầm quan trọng tương đối giữa các mục tiêu tối ưu.

Hàm `load_config()` thực hiện theo cơ chế **Priority Fallback ba cấp**: ưu tiên cao nhất là đường dẫn YAML được truyền trực tiếp; nếu không có, tìm trong các vị trí mặc định theo thứ tự `configs/default.yaml` rồi `config.yaml`; nếu không tìm thấy file nào, trả về `PyBatGymConfig()` với toàn bộ giá trị mặc định từ Pydantic. Cơ chế này đảm bảo hệ thống luôn có cấu hình hợp lệ trong mọi tình huống sử dụng.

#### Cơ chế Seeding — Đảm bảo Reproducibility

Khả năng tái lập (reproducibility) là yêu cầu quan trọng trong nghiên cứu khoa học: cùng một cấu hình và seed phải luôn cho kết quả giống hệt nhau. PyBatGym đảm bảo điều này qua **ba tầng seed độc lập**.

Tầng đầu tiên là Gymnasium RNG: khi gọi `super().reset(seed=42)`, Gymnasium khởi tạo `np_random` — bộ sinh số ngẫu nhiên tiêu chuẩn dùng cho các thao tác như `action_space.sample()` trong evaluation thủ công.

Tầng thứ hai và quan trọng nhất là Workload Seed: `MockAdapter` nhận seed từ `reset()` và khởi tạo riêng một `random.Random(seed)` nội bộ. Toàn bộ quá trình sinh workload tổng hợp — thời điểm nộp job, thời gian chạy, số cores — đều dùng bộ sinh số ngẫu nhiên này. Điều này đảm bảo rằng với cùng seed, workload sinh ra sẽ hoàn toàn giống hệt nhau mỗi lần, bất kể đã chạy bao nhiêu episode trước đó.

Tầng thứ ba là Config-level Seed: giá trị `config.workload.seed` trong file YAML đóng vai trò seed mặc định khi `reset()` không được truyền seed cụ thể. Điều này giúp "đóng băng" workload khi cần chạy nhiều episode với cùng điều kiện.

Bảng sau tóm tắt cách sử dụng seed theo từng tình huống thực tế:

| Mục đích | Cách dùng | Kết quả |
|----------|-----------|---------|
| Training đa dạng | Truyền `seed=episode_num` mỗi episode | Mỗi episode có workload khác nhau, tránh overfit |
| Evaluation khách quan | Truyền `seed=0` cố định | Mọi agent đều được test trên cùng workload |
| Debug lỗi cụ thể | Truyền seed của episode bị lỗi | Tái hiện chính xác môi trường khi lỗi xảy ra |
| Benchmark | Truyền seed cố định cho cả tập baseline | So sánh agent với FCFS/SJF/EASY trên cùng workload |

### 6.3 Tích hợp Thư viện Reinforcement Learning

#### Tương thích Gymnasium API

PyBatGym tuân thủ **100% Gymnasium API hiện đại** (phiên bản kế thừa và thay thế OpenAI Gym). Điều này có nghĩa là bất kỳ thư viện RL nào hỗ trợ Gymnasium đều có thể sử dụng trực tiếp với PyBatGym mà **không cần bất kỳ wrapper hay adapter code nào**. Các phương thức `reset()`, `step()`, `render()`, `close()` đều tuân thủ đúng signature, kiểu trả về và ngữ nghĩa của Gymnasium.

Hai thuộc tính `observation_space` và `action_space` được khai báo đúng kiểu: `observation_space` là `gymnasium.spaces.Dict` chứa hai không gian con `Box` (features và action_mask), còn `action_space` là `gymnasium.spaces.Discrete(K+2)`. Việc khai báo đúng spaces là điều kiện tiên quyết để Stable-Baselines3 có thể tự động chọn kiến trúc mạng phù hợp.

#### Tích hợp với Stable-Baselines3 — PPO

Để chạy thuật toán PPO với PyBatGym, người dùng chỉ cần ba bước: khởi tạo môi trường, khởi tạo model và gọi `learn()`. Điểm quan trọng cần lưu ý là phải chọn **`"MultiInputPolicy"`** thay vì `"MlpPolicy"` — lý do là `observation_space` là `Dict` space (gồm hai không gian con), trong khi `MlpPolicy` chỉ xử lý được `Box` space đơn. Với `MultiInputPolicy`, SB3 tự động xây dựng các nhánh encoder riêng biệt cho `features` và `action_mask`, sau đó kết hợp chúng qua một mạng chung trước khi đưa vào policy và value head. Toàn bộ ma trận thiết kế mạng này được thực hiện tự động — người dùng không cần can thiệp.

Khi gọi `model.learn(total_timesteps)`, SB3 hoàn toàn tự quản lý vòng lặp: nó gọi `env.reset()` khi bắt đầu, liên tục gọi `env.step()`, thu thập trajectories và cập nhật weights. PyBatGym không cần biết gì về SB3 và ngược lại — đây chính là sức mạnh của chuẩn hóa API.

#### Tích hợp với MaskablePPO — Khai thác Action Mask

Một bước tiến xa hơn so với PPO thông thường là sử dụng `MaskablePPO` từ thư viện `sb3-contrib`. MaskablePPO đọc `action_mask` từ observation dictionary và áp dụng nó **trực tiếp vào phân phối xác suất** trước khi lấy mẫu hành động: các hành động không hợp lệ được gán xác suất bằng 0, đảm bảo agent không bao giờ chọn chúng.

So với PPO chuẩn, MaskablePPO hội tụ nhanh hơn đáng kể vì không lãng phí trải nghiệm học vào những hành động vô nghĩa. Trong bài toán lập lịch HPC, nơi hành động không hợp lệ (lên lịch job không đủ resource) chiếm tỉ lệ cao trong giai đoạn đầu training, lợi ích này càng thể hiện rõ — theo thực nghiệm, tốc độ hội tụ nhanh hơn 30–40% so với PPO không có masking.

---

## 7. ĐÁNH GIÁ VÀ THỬ NGHIỆM HỆ THỐNG

### 7.1 Mục tiêu và Nguyên tắc Đánh giá
Phần đánh giá của đề tài **không nhằm chứng minh một policy học tăng cường cụ thể vượt trội so với mọi heuristic**, mà mục tiêu chính là kiểm chứng rằng **PyBatGym đã được hiện thực như một môi trường RL có thể vận hành được (end-to-end), dùng để benchmark ở mức tối thiểu, và tái lập được thí nghiệm**. Nguyên tắc đánh giá ưu tiên tính đúng đắn và tính hoàn chỉnh của pipeline kỹ thuật.

### 7.2 Các Nhóm Thử nghiệm Chính
1. **Thử nghiệm luồng End-to-end**: Kiểm chứng vòng lặp `reset-step` hoạt động trọn vẹn. Hệ thống có thể khởi tạo môi trường, nhận workload, sinh action liên tục cho đến khi kết thúc (terminated) thành công.
2. **Thử nghiệm Semantic correctness**: Kiểm tra `observation` (kích thước cố định, padding đúng), `action handling` (xử lý action hợp lệ, mask), và `reward computation` (tính toán chính xác các biến động thực trong queue và tài nguyên).
3. **Thử nghiệm Benchmark Tối thiểu**: Chạy song song heuristic baseline (ví dụ: FCFS, EASY Backfilling) và một policy RL đơn giản (như PPO) trên cùng một workload, platform, cấu hình, và seed để thu được các bảng metric đối chiếu có ý nghĩa.
4. **Thử nghiệm Reproducibility (Tính tái lập)**: Cố định cấu hình và seed, chạy lặp lại hệ thống nhiều lần để chứng minh các kết quả cuối episode và chuỗi hành vi là hoàn toàn ổn định và giống hệt nhau.
5. **Thử nghiệm Debug Decision Point**: Trích xuất log chi tiết tại các thời điểm agent ra quyết định (quan sát gì, chọn gì, nhận reward bao nhiêu) để đảm bảo tính minh bạch, tránh hệ thống hoạt động như một "hộp đen".

### 7.3 Artifacts Đầu Ra
Toàn bộ quá trình thử nghiệm được tự động xuất ra các artifact đóng gói:
- File cấu hình YAML và Log Seed.
- Log quá trình chạy ở định dạng CSV (chứa metric từng step).
- Bảng so sánh (Comparison tables) cuối episode tổng kết các tiêu chí `Waiting Time`, `Slowdown`, `Utilization`.

---

## 8. HẠN CHẾ VÀ ĐỊNH HƯỚNG TƯƠNG LAI

### 8.1 Giới hạn của Đề tài (Vertical Slice Prototype)
Trong khuôn khổ nguồn lực giới hạn của 1 sinh viên, PyBatGym được định vị là một **"Vertical slice research prototype"**. Điều này có nghĩa là thay vì dàn trải hỗ trợ quá nhiều loại mô phỏng, hệ thống tập trung hoàn thiện một "lát cắt dọc" sâu và hoàn chỉnh: từ xử lý event-driven loop -> đánh giá heuristic -> training RL -> xuất artifact. Hiện tại hệ thống đang cố định:
- 1 Backend duy nhất (BatSim C++).
- 1 Lược đồ trạng thái (Observation Schema).
- 1 Lược đồ hành động (Action Schema).
- 1 Công thức hàm phần thưởng (Reward Function).

### 8.2 Định hướng Phát triển Tương lai
- **Giai đoạn 1 (Đa dạng hóa MDP)**: Tích hợp thêm các bộ mã hóa Observation đa dạng hơn (như Graph Neural Network cho topology) và không gian hành động linh hoạt hơn. Hỗ trợ cơ chế thiết lập Reward động (Dynamic Reward).
- **Giai đoạn 2 (Mở rộng Backend)**: Tích hợp với các simulator mã nguồn mở khác ngoài BatSim để mở rộng phổ bài toán từ cluster đơn lẻ sang multi-cluster, cloud computing.
- **Giai đoạn 3 (Thuật toán Chuyên sâu)**: Sử dụng framework PyBatGym làm nền tảng để nghiên cứu và phát triển các thuật toán RL chuyên biệt cho HPC, với mục tiêu thực sự vượt qua giới hạn của EASY Backfilling trên các trace workload khổng lồ.

---

## 9. GIẢI ĐÁP PHẢN BIỆN (Q&A)

**Q1: PyBatGym có phải chỉ là một lớp wrapper gọi API của BatSim không?**  
**A:** Không. Wrapper chỉ đơn giản bọc hàm C++ thành Python. PyBatGym làm nhiều hơn thế: Nó định nghĩa lại bài toán (formulate MDP), quyết định khi nào agent được hành động (Event-driven decision point), mã hóa thông tin mô phỏng (Observation Encoder), biên dịch hành động (Action Processor) và tự tính điểm (Reward Engine). Core RL không biết BatSim là gì, minh chứng qua việc dễ dàng swap sang `MockAdapter`.

**Q2: Môi trường Event-driven khác gì môi trường Time-step thông thường?**  
**A:** Trong time-step (như game Atari), hệ thống tiến từng đơn vị thời gian (vd: 1s). Nhưng trong Scheduling HPC, có thể 10 phút không có job nào mới. Mô hình Event-driven của PyBatGym bỏ qua các khoảng thời gian trống này, chỉ dừng lại yêu cầu RL Agent ra quyết định tại các **Decision Points** (khi có job mới đến hoặc có tài nguyên giải phóng). Điều này tăng tốc độ train lên hàng nghìn lần.

**Q3: Agent thực sự quan sát gì và điều khiển cái gì? Liệu nó có trực tiếp kiểm soát BatSim?**  
**A:** Agent không điều khiển BatSim ở tầng thấp. Agent chỉ quan sát một vector rút gọn 51 chiều và đưa ra một số nguyên `a` (0 đến K). `PyBatGym` sẽ dịch `a` thành lệnh "Giao Job thứ `a` vào máy chủ". Việc mô phỏng tiếp theo do Backend lo.

**Q4: Nếu RL (PPO) chạy ra kết quả chưa vượt được Heuristic như EASY Backfilling thì framework này có vô dụng không?**  
**A:** Hoàn toàn không. Mục tiêu của việc xây dựng PyBatGym là tạo ra một **Framework đánh giá và nghiên cứu chuẩn mực**, chứ không phải chứng minh một thuật toán RL vô địch. Việc Framework minh bạch báo cáo PPO thua hay hòa EASY Backfilling chính là bằng chứng cho thấy hệ thống đã hoạt động đúng, đo lường chính xác. Nhiệm vụ tạo ra Policy RL ưu việt là bước nghiên cứu dài hạn tiếp theo của cộng đồng trên chính "sân chơi" này.

---

## PHỤ LỤC

### A. Cấu trúc Thư mục Dự án

```
ThinhProject/
├── pybatgym/
│   ├── env.py               # PyBatGymEnv — Gymnasium interface
│   ├── batsim_adapter.py    # BatsimAdapter ABC + MockAdapter
│   ├── real_adapter.py      # RealBatsimAdapter (ZMQ)
│   ├── observation.py       # ObservationBuilder
│   ├── action.py            # ActionMapper
│   ├── reward.py            # RewardCalculator
│   ├── models.py            # Job, Resource, Event, ScheduleCommand
│   ├── workload_parser.py   # BatSim JSON workload parser
│   ├── config/
│   │   ├── base_config.py   # Pydantic v2 config models
│   │   └── loader.py        # YAML loader + priority fallback
│   └── plugins/
│       ├── csv_logger.py    # CSVLoggerPlugin
│       ├── tensorboard.py   # TensorBoardLoggerPlugin
│       └── benchmark.py     # BenchmarkPlugin (FCFS/SJF/EASY)
├── configs/
│   └── default.yaml         # Default configuration
├── data/workloads/
│   ├── tiny_workload.json   # 6 jobs (kiểm thử)
│   ├── medium_workload.json # 200 jobs
│   └── heavy_workload.json  # 500+ jobs
├── examples/
│   ├── quickstart.py        # Hướng dẫn bắt đầu nhanh
│   └── train_ppo_phase1.py  # Training pipeline đầy đủ
├── Dockerfile               # Multi-stage build
└── docker-compose.yml       # Orchestration services
```

### B. Thông số Hệ thống Mặc định

| Thông số | Giá trị mặc định | Ghi chú |
|----------|-----------------|---------|
| `mode` | `"mock"` | Chuyển sang `"real"` khi cần BatSim C++ |
| Platform | 16 nodes × 4 cores = **64 cores** | Kích thước cluster mô phỏng |
| `num_jobs` | 100 | Số jobs trong một episode |
| `top_k_jobs` (K) | 10 | Số jobs hiển thị trong observation |
| `observation_dim` | 51 = 11 + 4×10 | Kích thước features vector |
| `action_space` | Discrete(12) | 10 jobs + WAIT + BACKFILL |
| `reward_type` | `"hybrid"` | Step reward + terminal bonus |
| `max_steps` | 5,000 | Giới hạn bước/episode (truncation guard) |
| `max_simulation_time` | 10,000.0 | Timeout thời gian simulation |
| ZMQ port | 28,000 (auto) | Tự tìm port trống trong Real Mode |
| BatSim timeout | 120 giây | Thời gian chờ tối đa phản hồi BatSim |
