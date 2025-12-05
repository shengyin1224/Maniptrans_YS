"""
可视化和分析接触结果的工具脚本
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint


def load_contact_results(result_path):
    """加载接触分析结果"""
    with open(result_path, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_contact_timeline(results, object_idx=0, save_path=None):
    """
    绘制接触时间线：显示每一帧哪些手部点与物体接触
    
    Args:
        results: 接触分析结果
        object_idx: 要可视化的物体索引
        save_path: 保存图片的路径（可选）
    """
    num_frames = results['num_frames']
    num_hand_points = results['num_hand_points']
    
    # 创建接触矩阵：[num_frames, num_hand_points]
    contact_matrix = np.zeros((num_frames, num_hand_points))
    distance_matrix = np.full((num_frames, num_hand_points), np.nan)
    
    for frame in results['frames']:
        frame_idx = frame['frame_idx']
        obj_data = frame['objects'][object_idx]
        
        for contact in obj_data['contacts']:
            hand_idx = contact['hand_point_idx']
            contact_matrix[frame_idx, hand_idx] = 1
            distance_matrix[frame_idx, hand_idx] = contact['distance'] * 1000  # 转换为毫米
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 子图1：接触热图
    im1 = ax1.imshow(contact_matrix.T, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Hand Point Index')
    ax1.set_title(f'Contact Timeline - {results["object_names"][object_idx]}')
    ax1.set_yticks(range(num_hand_points))
    ax1.set_yticklabels([f"{i}:{results['hand_point_names'][i]}" for i in range(num_hand_points)], fontsize=6)
    plt.colorbar(im1, ax=ax1, label='Contact (1=yes, 0=no)')
    
    # 子图2：接触距离热图
    im2 = ax2.imshow(distance_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=5)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Hand Point Index')
    ax2.set_title(f'Contact Distance (mm) - {results["object_names"][object_idx]}')
    ax2.set_yticks(range(num_hand_points))
    ax2.set_yticklabels([f"{i}:{results['hand_point_names'][i]}" for i in range(num_hand_points)], fontsize=6)
    plt.colorbar(im2, ax=ax2, label='Distance (mm)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        cprint(f"Figure saved to: {save_path}", "green")
    
    plt.show()


def plot_contact_statistics(results, save_path=None):
    """
    绘制接触统计图
    
    Args:
        results: 接触分析结果
        save_path: 保存图片的路径（可选）
    """
    num_frames = results['num_frames']
    num_objects = results['num_objects']
    num_hand_points = results['num_hand_points']
    
    # 统计每个手部点的接触次数
    hand_contact_counts = np.zeros(num_hand_points)
    
    # 统计每个物体的接触帧数
    object_contact_frames = np.zeros(num_objects)
    
    for frame in results['frames']:
        for obj_idx, obj_data in enumerate(frame['objects']):
            if obj_data['has_contact']:
                object_contact_frames[obj_idx] += 1
            
            for contact in obj_data['contacts']:
                hand_contact_counts[contact['hand_point_idx']] += 1
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1：手部点接触频率
    bars1 = ax1.bar(range(num_hand_points), hand_contact_counts)
    ax1.set_xlabel('Hand Point')
    ax1.set_ylabel('Total Contacts')
    ax1.set_title('Contact Frequency by Hand Point')
    ax1.set_xticks(range(num_hand_points))
    ax1.set_xticklabels([f"{i}\n{results['hand_point_names'][i]}" for i in range(num_hand_points)], 
                        rotation=45, ha='right', fontsize=7)
    ax1.grid(axis='y', alpha=0.3)
    
    # 标注数值
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=7)
    
    # 子图2：物体接触帧占比
    bars2 = ax2.bar(range(num_objects), object_contact_frames / num_frames * 100)
    ax2.set_xlabel('Object')
    ax2.set_ylabel('Contact Frame Percentage (%)')
    ax2.set_title('Contact Coverage by Object')
    ax2.set_xticks(range(num_objects))
    ax2.set_xticklabels(results['object_names'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 标注数值
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        cprint(f"Figure saved to: {save_path}", "green")
    
    plt.show()


def export_contacts_to_csv(results, output_path):
    """
    将接触数据导出为CSV格式，便于进一步分析
    
    Args:
        results: 接触分析结果
        output_path: CSV文件保存路径
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            'frame_idx',
            'object_idx',
            'object_name',
            'hand_point_idx',
            'hand_point_name',
            'hand_point_x',
            'hand_point_y',
            'hand_point_z',
            'object_contact_x',
            'object_contact_y',
            'object_contact_z',
            'distance_mm'
        ])
        
        # 写入数据
        for frame in results['frames']:
            frame_idx = frame['frame_idx']
            
            for obj_data in frame['objects']:
                obj_idx = obj_data['object_idx']
                obj_name = obj_data['object_name']
                
                for contact in obj_data['contacts']:
                    hand_idx = contact['hand_point_idx']
                    hand_name = contact['hand_point_name']
                    hand_pos = contact['hand_point_pos']
                    obj_pos = contact['object_contact_pos']
                    dist = contact['distance'] * 1000  # 转换为毫米
                    
                    writer.writerow([
                        frame_idx,
                        obj_idx,
                        obj_name,
                        hand_idx,
                        hand_name,
                        hand_pos[0],
                        hand_pos[1],
                        hand_pos[2],
                        obj_pos[0],
                        obj_pos[1],
                        obj_pos[2],
                        dist
                    ])
    
    cprint(f"CSV exported to: {output_path}", "green")


def print_frame_details(results, frame_idx):
    """
    打印某一帧的详细接触信息
    
    Args:
        results: 接触分析结果
        frame_idx: 帧索引
    """
    frame = results['frames'][frame_idx]
    
    cprint(f"\n{'='*80}", "cyan")
    cprint(f"Frame {frame_idx} Contact Details", "cyan", attrs=['bold'])
    cprint(f"{'='*80}", "cyan")
    
    for obj_data in frame['objects']:
        obj_name = obj_data['object_name']
        num_contacts = obj_data['num_contacts']
        
        if num_contacts > 0:
            cprint(f"\n{obj_name}: {num_contacts} contact(s)", "yellow", attrs=['bold'])
            
            for contact in obj_data['contacts']:
                hand_name = contact['hand_point_name']
                dist_mm = contact['distance'] * 1000
                hand_pos = contact['hand_point_pos']
                obj_pos = contact['object_contact_pos']
                
                cprint(f"  • {hand_name:30s} | Distance: {dist_mm:5.2f} mm", "white")
                cprint(f"    Hand: ({hand_pos[0]:7.4f}, {hand_pos[1]:7.4f}, {hand_pos[2]:7.4f})", "white")
                cprint(f"    Obj:  ({obj_pos[0]:7.4f}, {obj_pos[1]:7.4f}, {obj_pos[2]:7.4f})", "white")
        else:
            cprint(f"\n{obj_name}: No contact", "white")
    
    cprint(f"\n{'='*80}\n", "cyan")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize hand-object contact results")
    parser.add_argument("result_path", type=str, help="Path to contact result pickle file")
    parser.add_argument("--timeline", action="store_true", help="Plot contact timeline")
    parser.add_argument("--statistics", action="store_true", help="Plot contact statistics")
    parser.add_argument("--export_csv", type=str, default=None, help="Export to CSV file")
    parser.add_argument("--frame_detail", type=int, default=None, help="Print detailed info for a specific frame")
    parser.add_argument("--object_idx", type=int, default=0, help="Object index for timeline plot")
    parser.add_argument("--save_fig", type=str, default=None, help="Save figure to path")
    
    args = parser.parse_args()
    
    # 加载结果
    cprint(f"Loading results from: {args.result_path}", "cyan")
    results = load_contact_results(args.result_path)
    
    cprint(f"Data: {results['data_idx']} ({results['dataset_type']})", "green")
    cprint(f"Frames: {results['num_frames']}, Objects: {results['num_objects']}, "
           f"Hand points: {results['num_hand_points']}", "green")
    
    # 执行可视化
    if args.timeline:
        plot_contact_timeline(results, args.object_idx, args.save_fig)
    
    if args.statistics:
        plot_contact_statistics(results, args.save_fig)
    
    if args.export_csv:
        export_contacts_to_csv(results, args.export_csv)
    
    if args.frame_detail is not None:
        print_frame_details(results, args.frame_detail)
    
    # 如果没有指定任何操作，显示统计摘要
    if not any([args.timeline, args.statistics, args.export_csv, args.frame_detail is not None]):
        from compute_hand_object_contacts import print_contact_summary
        print_contact_summary(results)

